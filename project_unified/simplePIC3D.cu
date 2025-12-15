#include <cassert>
#include <cstdlib>
//#include <filesystem>
#include <math.h>
#include "fields.hpp"
//#include "fields_cuda.hpp"
#include "init.hpp"
#include "io.hpp"
#include "parameters.hpp"
#include "particles.hpp"
#include "structures.hpp"
#include "util.hpp"


int main(int argc, char **argv)
{
  if (argc < 3) 
  {
    std::cerr << "Error -> how to correctly launch the program: make run ARGS=\"<blockX> <blockY>\" \n\n";
    return -1;
  }
  int blockX = atoi(argv[1]), blockY = atoi(argv[2]);

  // loading parameters from the config.cfg file
  Parameters p;
  loadParameters("config.cfg", &p);
  printParameters(&p);

  // declaring structures of arrays for grid, fields and particles    
  simu_grid* grid;
  simu_fields* fields;
  simu_particles** part; // ~ part[ns];

  // memory allocation
  allocateGrid(&grid, p.nx, p.ny, p.nz);
  allocateFields(&fields, p.ns, p.nCells, p.nNodes);
  allocateParticles(&part, p.ns, p.np);

  // memory allocation on the GPU
  allocateGrid_gpu(&grid, p.nx, p.ny, p.nz);
  allocateFields_gpu(&fields, p.ns, p.nCells, p.nNodes);
  allocateParticles_gpu(&part, p.ns, p.np);

  //allocateGrid_gpu_standard(&grid, p.nx, p.ny, p.nz);
  //allocateFields_gpu_standard(&fields, p.ns, p.nCells, p.nNodes);
  //allocateParticles_gpu_standard(&part, p.ns, p.np);

  double** harvest_array;
  double** theta_array;


  allocateRandomArrays(harvest_array, theta_array, p.ns, p.nx, p.ny, p.nz, p.npcx, p.npcy, p.npcz);

  for (int is = 0; is < p.ns; is++) {
      int totalParticles = p.nx * p.ny * p.nz * p.npcx[is] * p.npcy[is] * p.npcz[is];
      initRandomNumbers(harvest_array[is], theta_array[is], totalParticles * 2);
  }

  // initialization
  printf("Initializing simulation\n");

  // ora inizializziamo la simu_grid, perciò definiamo i valori della struct
  // (vedi structures.hpp per spiegazione), dove andiamo a posizionare ciascun 
  // nodo della griglia in una posizione specifica (vedi il commento nella funzione
  // initGrid, nel file init.hpp).
  initGrid(grid, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz);
  printf("post initGrid grid\n");
  if (string(p.init_case) == "twostreams") 
  {  
    initEMFieldsTwostreams(fields, p.B0x, p.B0y, p.B0z, p.qom, p.ns, p.nCells);
    printf("Initializing particles for two stream instability\n");
    for (int is = 0; is < p.ns; is++)
    {
      int totalParticles = p.nx * p.ny * p.nz * p.npcx[is] * p.npcy[is] * p.npcz[is];
      int threads = blockX * blockY;
      int blocks = (totalParticles + threads - 1) / threads;

      initTwoPartsKernel<<<blocks, threads>>>(
          part[is], grid, fields,
          p.npcx[is], p.npcy[is], p.npcz[is],
          p.nx, p.ny, p.nz, p.dx, p.dy, p.dz,
          p.qom[is], p.u0[is], p.v0[is], p.w0[is],
          p.uth0[is], p.vth0[is], p.wth0[is],
          p.npc[is], harvest_array[is], theta_array[is]
      );
      cudaDeviceSynchronize();
    }
  }
  else if (string(p.init_case) == "randomInit")
  {
    // inizializza i campi elettromagnetici (vedi initEMfields nel file init.hpp)
    initEMfields(fields, p.B0x, p.B0y, p.B0z, p.Amp, p.qom, p.Lx, p.ns, p.nx, p.ny, p.nz, p.dx);
    for (int is = 0; is < p.ns; is++)
    {
      int totalParticles = p.nx * p.ny * p.nz * p.npcx[is] * p.npcy[is] * p.npcz[is];
      int threads = blockX * blockY;
      int blocks = (totalParticles + threads - 1) / threads;

      maxwellianKernel<<<blocks, threads>>>(
          part[is], grid, fields,
          p.npcx[is], p.npcy[is], p.npcz[is],
          p.nx, p.ny, p.nz, p.dx, p.dy, p.dz,
          p.qom[is], p.u0[is], p.v0[is], p.w0[is],
          p.uth0[is], p.vth0[is], p.wth0[is],
          p.npc[is], harvest_array[is], theta_array[is]
      );
      cudaDeviceSynchronize();
    }
  }
  else
  {
    printf("Unable to find a valid configuration. Aborting...\n");
    exit(EXIT_FAILURE);
  }

  // output
  string out_dir_path = "out/";
  //std::filesystem::create_directory(out_dir_path);
  system("mkdir -p out");
  saveGlobalQuantities(fields, part, p.ns, p.np, p.qom, p.nCells, p.dx, p.dy, p.dz, 0, out_dir_path.c_str());
  saveFields(fields, 0, p.ns, p.nCells, out_dir_path.c_str());
  //for (int is = 0; is < ns; is++)
  //  saveParticles(part, is, 0, out_dir_path.c_str());    
  
  // main loop
  util::Timer clTimer;
  double *rhos_private = nullptr;
  for (int it = 1; it <= p.nsteps; it++)
  {   
    printf("\rRunning simulation - step %d", it); fflush(stdout);
    //memset((unsigned char *)fields->rhos, 0.0, sizeof(double) * p.ns * p.nCells); // <<< use cudaMemset >>>
    cudaMemset(fields->rhos, 0, sizeof(double) * p.ns * p.nCells);    // Replace memset with cudaMemset for GPU memory
    for (int is = 0; is < p.ns; is++) 
    {
        // Calcolo del numero di blocchi e dimensione di ciascun blocco
      int numThreads = blockX * blockY; // Numero di thread per blocco
      int numBlocks = (p.np[is] + numThreads - 1) / numThreads; // Arrotonda per assicurare che tutti i particelle siano gestite

      updateParticlePositionKernel<<<numBlocks, numThreads>>>(part[is]->rx, part[is]->ry, part[is]->rz,
                                                                 part[is]->vx, part[is]->vy, part[is]->vz,
                                                                 p.np[is], p.Lx, p.Ly, p.Lz, p.dt);
      cudaDeviceSynchronize();
      // Lancio del kernel
      //basta semplicemente scrivere particles2GridKernel al posto di particles2GridKernelAggregation
      particles2GridKernel<<<numBlocks, numThreads>>>(
          fields, grid, p.nx, p.ny, p.nz, 
          p.dx, p.dy, p.dz, p.invVOL,
          part, is, p.np[is]
      );
     /*
      int numParticlesPerThread = 32;
      particles2GridKernelCoarsening<<<numBlocks, numThreads>>>(
        fields, grid, p.nx, p.ny, p.nz, 
        p.dx, p.dy, p.dz, p.invVOL,
        part, is, p.np[is],
        numParticlesPerThread
      );
      */
      /*
      size_t size = p.nCells * numBlocks * sizeof(double);
      cudaMallocManaged(&rhos_private, size);
      cudaMemset(rhos_private, 0.0, size);

      particles2GridKernelPrivatized<<<numBlocks, numThreads>>>(
          part, grid, fields,
          p.nx, p.ny, p.nz,
          p.dx, p.dy, p.dz,
          p.invVOL,
          is,
          p.np[is],
          rhos_private,
          p.nCells,
          numBlocks
      );
      */
      cudaDeviceSynchronize();

      // cudaFree(rhos_private);
    }

    // field solver begin	
    //memset((unsigned char *)fields->rho_tot, 0.0, sizeof(double) * p.nCells);   // <<< use cudaMemset >>>
    cudaMemset(fields->rho_tot, 0, sizeof(double) * p.nCells);
    //computeRhoTot_gpu
    // computeRhoTot(fields->rho_tot, fields->rhos, p.ns, p.nx, p.ny, p.nz);       // <<< convert to a cuda kernel >>>
    dim3 threadsPerBlock(blockX, blockY, 1); // Adjust based on your GPU's capability
    dim3 numBlocks((p.nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                  (p.ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                  (p.nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Launch the kernel
    computeRhoTotKernel<<<numBlocks, threadsPerBlock>>>(fields->rho_tot, fields->rhos, p.ns, p.nx, p.ny, p.nz);

    // Synchronize to ensure the kernel finishes
    cudaDeviceSynchronize();
    //poisson_gpu
    poisson(fields->phi, fields->rho_tot, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz, blockX, blockY);  // <<< go to the function definition... >>>
    gradient(fields->Exn, fields->Eyn, fields->Ezn, fields->phi,                // <<< go to the function definition... >>>
             p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz , -1, blockX, blockY);  
    // field solver end
    
    // particle velocity
    for (int is = 0; is < p.ns; is++){
      int numThreads = blockX * blockY; // Numero di thread per blocco
      int numBlocks = (p.np[is] + numThreads - 1) / numThreads; // Arrotonda per assicurare che tutti le particelle siano gestite
      updateParticleVelocityKernel<<<numBlocks, numThreads>>>(
      part[is]->rx, part[is]->ry, part[is]->rz,
      part[is]->vx, part[is]->vy, part[is]->vz,
      fields->Exn, fields->Eyn, fields->Ezn,
      grid->nodeX, grid->nodeY, grid->nodeZ,
      p.nx, p.ny, p.nz,
      p.dx, p.dy, p.dz, p.invVOL,
      p.qom[is], p.dt, p.np[is]
      );
      cudaDeviceSynchronize();
    }
    // output to file
    if (it % p.global_save_freq == 0) saveGlobalQuantities(fields, part, p.ns, p.np, p.qom, p.nCells, p.dx, p.dy, p.dz, it, out_dir_path.c_str());
    if (it % p.fields_save_freq == 0) saveFields(fields, it, p.ns, p.nCells, out_dir_path.c_str());
    if (it % p.part_save_freq == 0) 
      for (int is = 0; is < p.ns; is++) saveParticles(part, is, p.np, it, out_dir_path.c_str());
  }  // main loop end
  //cudaDeviceSynchronize();
  double elapsedTime = static_cast<double>(clTimer.getTimeMilliseconds());
  std::cout << std::endl << "Simulation terminated." << std::endl;
  std::cout << "Simulation loop elapsed time: " << elapsedTime << " ms (corresponding to " << (elapsedTime / 1000.0) << " s)" << std::endl;

  // memory de-allocation
  //if (grid) freeGrid(&grid);
  //if (fields) freeFields(&fields);
  //if (part) freeParticles(&part, p.ns);

  // memory de-allocation
  if (grid) freeGrid_gpu(&grid);
  if (fields) freeFields_gpu(&fields);
  if (part) freeParticles_gpu(&part, p.ns);
  freeRandomArrays(harvest_array, theta_array, p);

  return 0;
}