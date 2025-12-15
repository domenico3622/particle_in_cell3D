#include <cassert>
#include <cstdlib>
//#include <filesystem>

#include "fields.hpp"
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

  // Load parameters
  Parameters p;
  loadParameters("config.cfg", &p);
  printParameters(&p);

  simu_grid* grid;
  simu_fields* fields;
  simu_particles** part;

  double *d_nodeX, *d_nodeY, *d_nodeZ;
  double *d_Exn, *d_Eyn, *d_Ezn;
  double *d_rho_tot, *d_rhos;
  double *d_phi;
  double *d_rx, *d_ry, *d_rz;
  double *d_vx, *d_vy, *d_vz;
  double *d_q;
  int *d_ID;

  double **harvest_array, **theta_array;
  double *d_harvest_array, *d_theta_array;

  allocateGridStandard(&grid, &d_nodeX, &d_nodeY, &d_nodeZ, p.nx, p.ny, p.nz);
  allocateFieldsStandard(&fields, &d_phi, &d_rho_tot, &d_Exn, &d_Eyn, &d_Ezn, &d_rhos, p.ns, p.nCells, p.nNodes);
  allocateParticlesStandard(&part, &d_rx, &d_ry, &d_rz, &d_vx, &d_vy, &d_vz, &d_q, &d_ID, p.ns, p.np);
  allocateRandomArraysStandard(harvest_array, theta_array, d_harvest_array, d_theta_array, p.ns, p.nx, p.ny, p.nz, p.npcx, p.npcy, p.npcz);

  // Memory for poissonCUFFT
  double *d_phi_poisson, *d_rho_poisson;
  cudaMalloc(&d_phi_poisson, sizeof(double) * p.nCells);
  cudaMalloc(&d_rho_poisson, sizeof(double) * p.nCells);

  // Memory for gradientCUFFT
  double *d_gradX, *d_gradY, *d_gradZ, *d_phi_gradient;
  cudaMalloc(&d_gradX, sizeof(double) * p.nCells);
  cudaMalloc(&d_gradY, sizeof(double) * p.nCells);
  cudaMalloc(&d_gradZ, sizeof(double) * p.nCells);
  cudaMalloc(&d_phi_gradient, sizeof(double) * p.nCells);

  printf("Initializing simulation\n");
  initGrid(grid, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz);

  int totalCells = p.nx * p.ny * p.nz; 
  int blockTotalSize = blockX * blockY;
  dim3 blockSize = blockTotalSize;
  dim3 gridSize = ((totalCells + blockSize.x - 1) / blockSize.x);
  const int PARTICLES_PER_THREAD = 32; 
  int sharedMemSize = blockTotalSize * sizeof(double); 

  // Copy grid coordinates to GPU
  cudaMemcpy(d_nodeX, grid->nodeX, (p.nx+1) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nodeY, grid->nodeY, (p.ny+1) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nodeZ, grid->nodeZ, (p.nz+1) * sizeof(double), cudaMemcpyHostToDevice);

  if (string(p.init_case) == "twostreams") 
  {  
    initEMFieldsTwostreams(fields, p.B0x, p.B0y, p.B0z, p.qom, p.ns, p.nCells);
    cudaMemcpy(d_rhos, fields->rhos, p.ns * p.nCells * sizeof(double), cudaMemcpyHostToDevice);

    for (int is = 0; is < p.ns; is++){
      initRandomNumbersStandard(harvest_array[is], theta_array[is], is, p.npc, p.npcx, p.npcy, p.npcz,
                                p.nx, p.ny, p.nz);
      int offsetParticles = is * p.np[is];
      int offsetFields = 2 * p.npcx[is] * p.npcy[is] * p.npcz[is] * p.nx * p.ny * p.nz;

      cudaMemcpy(d_harvest_array + offsetFields*is, harvest_array[is], offsetFields * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_theta_array + offsetFields*is, theta_array[is], offsetFields * sizeof(double), cudaMemcpyHostToDevice);      
      cudaMemcpy(d_q + offsetParticles, part[is]->q, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vx + offsetParticles, part[is]->vx, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vy + offsetParticles, part[is]->vy, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vz + offsetParticles, part[is]->vz, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_rx + offsetParticles, part[is]->rx, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ry + offsetParticles, part[is]->ry, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_rz + offsetParticles, part[is]->rz, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ID + offsetParticles, part[is]->ID, p.np[is] * sizeof(int), cudaMemcpyHostToDevice);

      initPartTwostreamsKernel<<<gridSize, blockSize>>>(
        d_rx + offsetParticles, d_ry + offsetParticles, d_rz + offsetParticles, d_q + offsetParticles,
        d_vx + offsetParticles, d_vy + offsetParticles, d_vz + offsetParticles, d_ID + offsetParticles,
        d_nodeX, d_nodeY, d_nodeZ, d_rhos + is * p.nCells,
        p.npcx[is], p.npcy[is], p.npcz[is], p.nx, p.ny, p.nz, p.dx, p.dy, p.dz,
        p.qom[is], p.u0[is], p.v0[is], p.w0[is], p.uth0[is], p.vth0[is], p.wth0[is], p.npc[is],
        d_harvest_array + offsetFields*is, d_theta_array + offsetFields*is
      );
      cudaDeviceSynchronize();
    }
  }
  else if (string(p.init_case) == "randomInit")
  {
    initEMfields(fields, p.B0x, p.B0y, p.B0z, p.Amp, p.qom, p.Lx, p.ns, p.nx, p.ny, p.nz, p.dx);
    cudaMemcpy(d_rhos, fields->rhos, p.ns * p.nCells * sizeof(double), cudaMemcpyHostToDevice);

    for (int is = 0; is < p.ns; is++)
    {
      initRandomNumbersStandard(harvest_array[is], theta_array[is], is, p.npc, p.npcx, p.npcy, p.npcz,
                                p.nx, p.ny, p.nz);
      int offsetParticles = is * p.np[is];
      int offsetFields = 2 * p.npcx[is] * p.npcy[is] * p.npcz[is] * p.nx * p.ny * p.nz;

      cudaMemcpy(d_harvest_array + offsetFields*is, harvest_array[is], offsetFields * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_theta_array + offsetFields*is, theta_array[is], offsetFields * sizeof(double), cudaMemcpyHostToDevice);      
      cudaMemcpy(d_q + offsetParticles, part[is]->q, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vx + offsetParticles, part[is]->vx, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vy + offsetParticles, part[is]->vy, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vz + offsetParticles, part[is]->vz, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_rx + offsetParticles, part[is]->rx, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ry + offsetParticles, part[is]->ry, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_rz + offsetParticles, part[is]->rz, p.np[is] * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_ID + offsetParticles, part[is]->ID, p.np[is] * sizeof(int), cudaMemcpyHostToDevice);

      maxwellianKernelStandard<<<gridSize, blockSize>>>(
        d_rx + offsetParticles, d_ry + offsetParticles, d_rz + offsetParticles, d_q + offsetParticles,
        d_vx + offsetParticles, d_vy + offsetParticles, d_vz + offsetParticles, d_ID + offsetParticles,
        d_nodeX, d_nodeY, d_nodeZ, d_rhos,
        p.npcx[is], p.npcy[is], p.npcz[is], p.nx, p.ny, p.nz, p.dx, p.dy, p.dz,
        p.qom[is], p.u0[is], p.v0[is], p.w0[is],
        p.uth0[is], p.vth0[is], p.wth0[is], p.npc[is],
        d_harvest_array + offsetFields*is, d_theta_array + offsetFields*is
      );
      cudaDeviceSynchronize();
    }
  }
  else
  {
    printf("Unable to find a valid configuration. Aborting...\n");
    exit(EXIT_FAILURE);
  }

  string out_dir_path = "out/";
  system("mkdir -p out");

  // Instead of copying every iteration, we copy once here for initial output
  cudaMemcpy(fields->rho_tot, d_rho_tot, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(fields->rhos, d_rhos, p.ns * p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(fields->Exn, d_Exn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(fields->Eyn, d_Eyn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(fields->Ezn, d_Ezn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
  for (int is = 0; is < p.ns; is++){
    int offsetParticles = is * p.np[is];
    cudaMemcpy(part[is]->rx, d_rx + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->ry, d_ry + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->rz, d_rz + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->vx, d_vx + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->vy, d_vy + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->vz, d_vz + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->q, d_q + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(part[is]->ID, d_ID + offsetParticles, p.np[is] * sizeof(int), cudaMemcpyDeviceToHost);
  }

  saveGlobalQuantities(fields, part, p.ns, p.np, p.qom, p.nCells, p.dx, p.dy, p.dz, 0, out_dir_path.c_str());
  saveFields(fields, 0, p.ns, p.nCells, out_dir_path.c_str());

  util::Timer clTimer;

  double *rhos_private = nullptr;

  for (int it = 1; it <= p.nsteps; it++)
  {   
    printf("\rRunning simulation - step %d", it); fflush(stdout);

    cudaMemset(d_rhos, 0, sizeof(double) * p.ns * p.nCells);
    for (int is = 0; is < p.ns; is++) 
    {
      int offsetParticles = is * p.np[is];
      int blocksPerGrid = (p.np[is] + blockTotalSize - 1) / blockTotalSize;
      // size_t sizePrivate = (size_t)blocksPerGrid * p.nCells * sizeof(double);
      // cudaMalloc(&rhos_private, sizePrivate);
      // cudaMemset(rhos_private, 0, sizePrivate);

      updateParticlePositionKernel<<<blocksPerGrid, blockSize>>>(
        d_rx + offsetParticles, d_ry + offsetParticles, d_rz + offsetParticles,
        d_vx + offsetParticles, d_vy + offsetParticles, d_vz + offsetParticles,
        p.np[is], p.Lx, p.Ly, p.Lz, p.dt
      );
      cudaDeviceSynchronize();
      /*
      // Coarsening for particles2Grid
      blocksPerGrid = (p.np[is] + (blockTotalSize * PARTICLES_PER_THREAD) - 1) / (blockTotalSize * PARTICLES_PER_THREAD);
      particles2GridKernelCoarsening<<<blocksPerGrid, blockTotalSize>>>(
          d_rhos, d_nodeX, d_nodeY, d_nodeZ,
          d_rx + offsetParticles, d_ry + offsetParticles, d_rz + offsetParticles, 
          d_q + offsetParticles,  
          p.nx, p.ny, p.nz, 
          p.dx, p.dy, p.dz, p.invVOL,
          is, p.np[is], PARTICLES_PER_THREAD
      );
      */
      particles2GridKernel<<<blocksPerGrid, blockTotalSize, sharedMemSize>>>(
          d_rhos, d_nodeX, d_nodeY, d_nodeZ,
          d_rx + offsetParticles, d_ry + offsetParticles, d_rz + offsetParticles, 
          d_q + offsetParticles,  
          p.nx, p.ny, p.nz, 
          p.dx, p.dy, p.dz, p.invVOL,
          is, p.np[is]
      );
      /*
      particles2GridKernelPrivatized<<<blocksPerGrid, blockTotalSize>>>(
          d_rhos,
          d_nodeX, d_nodeY, d_nodeZ,
          d_rx + offsetParticles, d_ry + offsetParticles, d_rz + offsetParticles,
          d_q  + offsetParticles,
          p.nx, p.ny, p.nz,
          p.dx, p.dy, p.dz,
          p.invVOL,
          is,
          p.np[is],
          rhos_private,
          p.nCells,
          blocksPerGrid
      );
      */
     /*
      particles2GridKernelAggregation<<<blocksPerGrid, blockTotalSize>>>(
          d_rhos,
          d_nodeX, d_nodeY, d_nodeZ,
          d_rx + offsetParticles, d_ry + offsetParticles, d_rz + offsetParticles,
          d_q  + offsetParticles,
          p.nx, p.ny, p.nz,
          p.dx, p.dy, p.dz,
          p.invVOL,
          is,
          p.np[is]
      );
      */
      cudaDeviceSynchronize();
      // cudaFree(rhos_private);
      // rhos_private = nullptr;
    }

    cudaMemset(d_rho_tot, 0, sizeof(double) * p.nCells);
    computeRhoTotKernel<<<gridSize, blockSize>>>(d_rho_tot, d_rhos, p.ns, p.nx, p.ny, p.nz);
    cudaDeviceSynchronize();

    cudaMemcpy(d_rho_poisson, d_rho_tot, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);
    poissonCUFFT(d_phi_poisson, d_rho_poisson, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz, blockX, blockY);

    cudaMemcpy(d_phi_gradient, d_phi_poisson, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);
    gradientCUFFT(d_gradX, d_gradY, d_gradZ, d_phi_gradient, p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz, -1, blockX, blockY);

    cudaMemcpy(d_Exn, d_gradX, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_Eyn, d_gradY, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_Ezn, d_gradZ, p.nCells * sizeof(double), cudaMemcpyDeviceToDevice);

    for (int is = 0; is < p.ns; is++){
      int offsetParticles = is * p.np[is];
      int blocksPerGrid = (p.np[is] + blockTotalSize - 1) / blockTotalSize;

      updateParticleVelocityKernel<<<blocksPerGrid, blockTotalSize>>>(
        d_rx + offsetParticles,
        d_ry + offsetParticles, d_rz + offsetParticles,
        d_vx + offsetParticles, d_vy + offsetParticles, d_vz + offsetParticles,
        d_Exn, d_Eyn, d_Ezn,
        d_nodeX, d_nodeY, d_nodeZ,
        p.nx, p.ny, p.nz,
        p.dx, p.dy, p.dz, p.invVOL,
        p.qom[is], p.dt, p.np[is]
      );
      cudaDeviceSynchronize();
    }

    // Perform device-to-host copies only at saving intervals
    if (it % p.global_save_freq == 0 || it % p.fields_save_freq == 0 || it % p.part_save_freq == 0) {
      // Copy fields back to host
      cudaMemcpy(fields->rho_tot, d_rho_tot, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(fields->rhos, d_rhos, p.ns * p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(fields->Exn, d_Exn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(fields->Eyn, d_Eyn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(fields->Ezn, d_Ezn, p.nCells * sizeof(double), cudaMemcpyDeviceToHost);

      // Copy particle data back to host
      for (int is = 0; is < p.ns; is++){
        int offsetParticles = is * p.np[is];
        cudaMemcpy(part[is]->rx, d_rx + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->ry, d_ry + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->rz, d_rz + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->vx, d_vx + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->vy, d_vy + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->vz, d_vz + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->q, d_q + offsetParticles, p.np[is] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(part[is]->ID, d_ID + offsetParticles, p.np[is] * sizeof(int), cudaMemcpyDeviceToHost);
      }

      // Save results
      if (it % p.global_save_freq == 0) 
        saveGlobalQuantities(fields, part, p.ns, p.np, p.qom, p.nCells, p.dx, p.dy, p.dz, it, out_dir_path.c_str());
      if (it % p.fields_save_freq == 0)
        saveFields(fields, it, p.ns, p.nCells, out_dir_path.c_str());
      if (it % p.part_save_freq == 0)
        for (int is = 0; is < p.ns; is++) saveParticles(part, is, p.np, it, out_dir_path.c_str());
    }

  } // end simulation loop

  double elapsedTime = static_cast<double>(clTimer.getTimeMilliseconds());
  std::cout << std::endl << "Simulation terminated." << std::endl;
  std::cout << "Simulation loop elapsed time: " << elapsedTime << " ms (corresponding to " << (elapsedTime / 1000.0) << " s)" << std::endl;

  // Free memory
  freeGrid(&grid, &d_nodeX, &d_nodeY, &d_nodeZ);
  freeFields(&fields, &d_phi, &d_rho_tot, &d_Exn, &d_Eyn, &d_Ezn, &d_rhos);
  freeParticles(&part, p.ns, &d_rx, &d_ry, &d_rz, &d_vx, &d_vy, &d_vz, &d_q, &d_ID);

  cudaFree(d_phi_poisson);
  cudaFree(d_rho_poisson);
  cudaFree(d_gradX);
  cudaFree(d_gradY);
  cudaFree(d_gradZ);
  cudaFree(d_phi_gradient);

  return 0;
}
