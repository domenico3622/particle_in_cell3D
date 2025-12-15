#ifndef INIT_HPP
#define INIT_HPP

#include <math.h>
#include "parameters.hpp"
#include "structures.hpp"
#include <cufft.h>
#include <cuComplex.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>

/*! Maxellian random velocity and uniform spatial distribution */

// ~~~~~~~~~~~~~~~ATTENZIONE~~~~~~~~~~~~~~~~
// nota che tutti questi array che abbiamo nella signature sono di dimensione NS_MAX: se vedi
// parameters.hpp, noterai che sta settando i valori solo per il numero effettivo di specie in 
// config.cfg, perciò avremo nel nostro caso le prime due posizioni inizializzate correttamente 
// e le restanti a zero
void maxwellian(
  simu_particles** part, // Array di strutture contenenti i dati di ogni particella (posizione, velocità, ecc.).
  int is,                // ID della specie (0 o 1, dato che abbiamo due specie nel nostro caso).
  int npc[],             // Numero totale di particelle per cella per ciascuna specie.
  int npcx[],            // Numero di particelle lungo x per cella per ciascuna specie.
  int npcy[],            // Numero di particelle lungo y per cella per ciascuna specie.
  int npcz[],            // Numero di particelle lungo z per cella per ciascuna specie.
  double qom[],          // Rapporto carica/massa per ciascuna specie.
  double u0[],           // Velocità media iniziale lungo x per ciascuna specie.
  double v0[],           // Velocità media iniziale lungo y per ciascuna specie.
  double w0[],           // Velocità media iniziale lungo z per ciascuna specie.
  double uth0[],         // Deviazione standard della velocità lungo x (distribuzione termica).
  double vth0[],         // Deviazione standard della velocità lungo y.
  double wth0[],         // Deviazione standard della velocità lungo z.
  simu_grid * grid,      // Struttura che rappresenta la griglia (posizione dei nodi, limiti, ecc.).
  int nx,                // Numero di macroparticelle lungo l'asse x.
  int ny,                // Numero di macroparticelle lungo l'asse y.
  int nz,                // Numero di macroparticelle lungo l'asse z.
  double dx,             // Passo spaziale lungo x (distanza tra due nodi consecutivi lungo x).
  double dy,             // Passo spaziale lungo y.
  double dz,             // Passo spaziale lungo z.
  simu_fields * EMf      // Struttura contenente i campi elettromagnetici e altre grandezze fisiche.
)
{
  double harvest;
  double prob, theta;

  int counter = 0;

  // triplo for per iterare tutte le macroparticelle.
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)     
      for (int k = 0; k < nz; k++)
    
        // triplo for per iterare su tutte le particelle contenute
        // in una macroparticella (per ciascuna specie).
        for (int ii = 0; ii < npcx[is]; ii++)
          for (int jj = 0; jj < npcy[is]; jj++)
            for (int kk = 0; kk < npcz[is]; kk++)
            {
              // rx ry rz
              part[is]->rx[counter] = (ii + .5) * (dx / npcx[is]) + grid->nodeX[i]; // x[i] = xstart + (xend-xstart)/2.0 + harvest1*((xend-xstart)/4.0)*cos(harvest2*2.0*M_PI);
              part[is]->ry[counter] = (jj + .5) * (dy / npcy[is]) + grid->nodeY[j];
              part[is]->rz[counter] = (kk + .5) * (dz / npcz[is]) + grid->nodeZ[k];
              // q
              part[is]->q[counter] = (qom[is] / fabs(qom[is])) * (fabs(EMf->rhos[is * (nx * ny * nz) + i] / npc[is]) * (dx * dy * dz));
              // vx
              harvest = rand() / (double)RAND_MAX;
              prob = sqrt(-2.0 * log(1.0 - .999999 * harvest));
              harvest = rand() / (double)RAND_MAX;
              theta = 2.0 * M_PI * harvest;
              part[is]->vx[counter] = u0[is] + uth0[is] * prob * cos(theta);
              // vy
              part[is]->vy[counter] = v0[is] + vth0[is] * prob * sin(theta);
              // wz
              harvest = rand() / (double)RAND_MAX;
              prob = sqrt(-2.0 * log(1.0 - .999999 * harvest));
              harvest = rand() / (double)RAND_MAX;
              theta = 2.0 * M_PI * harvest;
              part[is]->vz[counter] = w0[is] + wth0[is] * prob * cos(theta);
              // ID
              part[is]->ID[counter] = counter;
             
              counter++;
            }
}

void initPartTwostreams(
    simu_particles** part, 
    int is, 
    int npc[], 
    int npcx[], int npcy[], int npcz[], 
    double qom[], 
    double u0[], double v0[], double w0[], 
    double uth0[], double vth0[], double wth0[],
    simu_grid * grid, 
    int nx, int ny, int nz, 
    double dx, double dy, double dz,
    simu_fields * EMf
)
{
  double harvest;
  double prob, theta;
  
  int counter = 0;
  float sign = -1.0;
  
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)     
      for (int k = 0; k < nz; k++)

          for (int ii = 0; ii < npcx[is]; ii++)
            for (int jj = 0; jj < npcy[is]; jj++)
              for (int kk = 0; kk < npcz[is]; kk++)
              {
                // rx ry rz
                part[is]->rx[counter] = (ii + .5) * (dx / npcx[is]) + grid->nodeX[i]; // x[i] = xstart + (xend-xstart)/2.0 + harvest1*((xend-xstart)/4.0)*cos(harvest2*2.0*M_PI);
                part[is]->ry[counter] = (jj + .5) * (dy / npcy[is]) + grid->nodeY[j];
                part[is]->rz[counter] = (kk + .5) * (dz / npcz[is]) + grid->nodeZ[k];
                // q
                part[is]->q[counter] = (qom[is] / fabs(qom[is])) * (fabs(EMf->rhos[is * (nx * ny * nz) + i] / npc[is]) * (dx * dy * dz));
                // vx
                harvest = rand() / (double)RAND_MAX;
                prob = sqrt(-2.0 * log(1.0 - .999999 * harvest));
                harvest = rand() / (double)RAND_MAX;
                theta = 2.0 * M_PI * harvest;
                part[is]->vx[counter] = u0[is]*sign + uth0[is] * prob * cos(theta);
                // vy
                part[is]->vy[counter] = v0[is] + vth0[is] * prob * sin(theta);
                // wz
                harvest = rand() / (double)RAND_MAX;
                prob = sqrt(-2.0 * log(1.0 - .999999 * harvest));
                harvest = rand() / (double)RAND_MAX;
                theta = 2.0 * M_PI * harvest;
                part[is]->vz[counter] = w0[is] + wth0[is] * prob * cos(theta);
                // ID
                part[is]->ID[counter] = counter;

	              sign*=-1.0; //change sign at each particle to create two beams along x

                counter++;
              }
}


void initGrid(simu_grid * grid, double Lx, double Ly, double Lz, int nx, int ny, int nz)
{
  grid->minX = 0; grid->maxX = Lx;
  grid->minY = 0; grid->maxY = Ly;
  grid->minZ = 0; grid->maxZ = Lz;

  // qui stiamo facendo il rapporto tra la size totale della griglia e il numero di nodi,
  // per ogni dimensione, perciò ogni nodo verrà posizionato equidistante da ogni altro.
  double dx = Lx / nx;
  double dy = Ly / ny;
  double dz = Lz / nz;

  for (int i = 0; i < nx+1; i++)
    grid->nodeX[i] = i * dx;

  for (int i = 0; i < ny+1; i++)
    grid->nodeY[i] = i * dy;
  
  for (int i = 0; i < nz+1; i++)
    grid->nodeZ[i] = i * dz;
}

/*! initialize Magnetic and Electric Field with initial configuration */
void initEMfields(simu_fields * EMf, double B0x, double B0y, double B0z, double Amp, double qom[], double Lx, int ns, int nx, int ny, int nz, double dx)
{
  // itero per tutte le macroparticelle presenti nella simulazione.
  for (int i = 0; i < nx * ny * nz; i++)
  {
    // itero per tutte le specie (due nel nostro caso)
    // e per ciascuna secie definisco la densità di carica.
    for (int is = 0; is < ns; is++)
      EMf->rhos[is * nx * ny * nz + i] = (qom[is]/fabs(qom[is])) * (1 / (4 * M_PI));

    // inizializzo il vettore del campo magnetico.
    EMf->Bxn[i] = B0x;
    EMf->Byn[i] = B0y;
    EMf->Bzn[i] = B0z;
  }
 
  // se siamo nel caso 1D, inizializza il campo elettrico solo sull'asse x
  // altrimenti lo setta tutto a zero.
  if (ny == 1 && nz == 1)
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
            {
                EMf->Exn[i * nz * ny + j * nz + k] = Amp * (2 * M_PI / Lx) * cos(i * dx * 2 * M_PI / Lx);
                EMf->Eyn[i * nz * ny + j * nz + k] = 0.0;
                EMf->Ezn[i * nz * ny + j * nz + k] = 0.0;
            }
  else
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
            {
                EMf->Exn[i * nz * ny + j * nz + k] = 0.0;
                EMf->Eyn[i * nz * ny + j * nz + k] = 0.0;
                EMf->Ezn[i * nz * ny + j * nz + k] = 0.0;
            }
}

void initEMFieldsTwostreams(simu_fields * EMf, double B0x, double B0y, double B0z, double qom[], int ns, int nCells)
{
  for (int i = 0; i < nCells; i++)
  {
    for (int is = 0; is < ns; is++)
      EMf->rhos[is * nCells + i] = (qom[is]/fabs(qom[is])) * (1 / (4 * M_PI));

    EMf->Bxn[i] = B0x;
    EMf->Byn[i] = B0y;
    EMf->Bzn[i] = B0z;
    EMf->Exn[i] = 0.0;
    EMf->Eyn[i] = 0.0;
    EMf->Ezn[i] = 0.0;
  }
}



void allocateRandomArrays(double**& harvest_array, double**& theta_array, int ns, int nx, int ny, int nz, int* npcx, int* npcy, int* npcz) {
    cudaMallocManaged(&harvest_array, ns * sizeof(double*));
    cudaMallocManaged(&theta_array, ns * sizeof(double*));
    
    for (int is = 0; is < ns; is++) {
        int totalParticles = nx * ny * nz * npcx[is] * npcy[is] * npcz[is] * 2;
        cudaMallocManaged(&harvest_array[is], totalParticles * sizeof(double));
        cudaMallocManaged(&theta_array[is], totalParticles * sizeof(double));
    }
}


void initRandomNumbers(double* harvest, double* theta, int totalParticles) {
    for (int i = 0; i < totalParticles; i++) {
        harvest[i] = rand() / (double)RAND_MAX;
        theta[i] = 2.0 * M_PI * (rand() / (double)RAND_MAX);
    }
}
void freeRandomArrays(double** harvest_array, double** theta_array, Parameters& p) {
    for (int is = 0; is < p.ns; is++) {
        cudaFree(harvest_array[is]);
        cudaFree(theta_array[is]);
    }
    cudaFree(harvest_array);
    cudaFree(theta_array);
}

// ~~~~~~~~~~~~~~~~~~~~ KERNEL DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// In this case, the domain is all the microparticles in our simulation.
// since we know from the initial configuration that we have, in each
// macro-particle a 1D array, we can define the cuda blocks and grid this way as well.
__global__ void initTwoPartsKernel(
    simu_particles* part, 
    simu_grid* grid, 
    simu_fields* EMf,
    int npcx, int npcy, int npcz,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double qom, 
    double u0, double v0, double w0,
    double uth0, double vth0, double wth0, 
    int npc,
    double* harvest_array, double* theta_array
) 
{
    // we calculate the index of the element to be computed
    // by each thread, checking also out-of-bound
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalParticles = nx * ny * nz * npcx * npcy * npcz;
    if (threadId >= totalParticles) return;

    int particlesPerCell = npcx * npcy * npcz;

    /// cellIndex will be storing the index of the macro-particle where the
    // the micro-particle being managed by each thread belongs.
    //
    // particleInCell will be storing the index of the microparticle
    // being managed by the current thread in its current MACROparticle
    int cellIndex = threadId / particlesPerCell;
    int particleInCell = threadId % particlesPerCell;

    // we convert the index of the macro-particle to 3D indexes
    int i = cellIndex / (ny * nz);
    int j = (cellIndex / nz) % ny;
    int k = cellIndex % nz;

    // Convert the particle index to 3D indexes
    int ii = particleInCell / (npcy * npcz);
    int jj = (particleInCell / npcz) % npcy;
    int kk = particleInCell % npcz;

    int counter = threadId;

    int random_index = counter * 2;

    part->rx[counter] = (ii + 0.5) * (dx / npcx) + grid->nodeX[i];
    part->ry[counter] = (jj + 0.5) * (dy / npcy) + grid->nodeY[j];
    part->rz[counter] = (kk + 0.5) * (dz / npcz) + grid->nodeZ[k];

    part->q[counter] = (qom / fabs(qom)) *
                       (fabs(EMf->rhos[cellIndex] / npc) * (dx * dy * dz));

    double harvest = harvest_array[random_index];
    double prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
    double theta = theta_array[random_index];
    part->vx[counter] = u0 + uth0 * prob * cos(theta);

    part->vy[counter] = v0 + vth0 * prob * sin(theta);

    harvest = harvest_array[random_index + 1];
    prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
    theta = theta_array[random_index + 1];
    part->vz[counter] = w0 + wth0 * prob * cos(theta);

    part->ID[counter] = counter;
}

// similar explanation of initTwoPartsKernel (see above)
__global__ void maxwellianKernel(
    simu_particles* part, simu_grid* grid, simu_fields* EMf,
    int npcx, int npcy, int npcz,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double qom, double u0, double v0, double w0,
    double uth0, double vth0, double wth0, int npc,
    double* harvest_array, double* theta_array
) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalParticles = nx * ny * nz * npcx * npcy * npcz;

    if (threadId >= totalParticles) return;

    int particlesPerCell = npcx * npcy * npcz;
    int cellIndex = threadId / particlesPerCell;
    int particleInCell = threadId % particlesPerCell;

    int i = cellIndex / (ny * nz);
    int j = (cellIndex / nz) % ny;
    int k = cellIndex % nz;

    int ii = particleInCell / (npcy * npcz);
    int jj = (particleInCell / npcz) % npcy;
    int kk = particleInCell % npcz;

    int counter = threadId;
    int random_index = counter * 2;

    // Particle position
    part->rx[counter] = (ii + 0.5) * (dx / npcx) + grid->nodeX[i];
    part->ry[counter] = (jj + 0.5) * (dy / npcy) + grid->nodeY[j];
    part->rz[counter] = (kk + 0.5) * (dz / npcz) + grid->nodeZ[k];

    // Particle charge
    part->q[counter] = (qom / fabs(qom)) *
                       (fabs(EMf->rhos[cellIndex] / npc) * (dx * dy * dz));

    // Particle velocity (vx, vy, vz)
    double harvest = harvest_array[random_index];
    double prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
    double theta = theta_array[random_index];
    part->vx[counter] = u0 + uth0 * prob * cos(theta);
    part->vy[counter] = v0 + vth0 * prob * sin(theta);

    harvest = harvest_array[random_index + 1];
    prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
    theta = theta_array[random_index + 1];
    part->vz[counter] = w0 + wth0 * prob * cos(theta);

    // Particle ID
    part->ID[counter] = counter;
}


// qui non stiamo più mappando i thread per ogni microparticella
// bensì per MACROparticella, infatti nel main definiamo la blockSize 
// e la gridSize come dim3, e per questo calcoliamo l'indice linearizzato per
// la matrice 3D delle MACROparticelle dentro cellIndex
__global__ void computeRhoTotKernel(double* rho_tot, double* rhos, int ns, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure we are within bounds
    if (i < nx && j < ny && k < nz) {
        int cellIndex = i * ny * nz + j * nz + k;

        // Accumulate contributions from all species
        double rhoSum = 0.0;
        for (int is = 0; is < ns; is++) {
            rhoSum += -4.0 * M_PI * rhos[is * nx * ny * nz + cellIndex];
        }
        rho_tot[cellIndex] = rhoSum;
    }
}



// Kernel to copy rho into in (if rho and in are both on GPU)
__global__ void copyRhoKernel(double* in, const double* rho, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        in[tid] = rho[tid];
    }
}

// questo kernel serve a normalizzare i valori contenuti in phi
// (li dividiamo tutti per la stessa quantità)
// qui il mapping dei thread è unidimensionale
__global__ void normalizeKernel(double* phi, int Nx, int Ny, int Nz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = Nx * Ny * Nz;
    if (tid < size) {
        phi[tid] /= (double)(Nx * Ny * Nz);
    }
}

// Kernel per risolvere l'equazione di Poisson nello spazio di Fourier.
// Divide ogni elemento di `out` per il fattore corrispondente al quadrato della norma del vettore d'onda.
__global__ void solvePoissonFourierKernel(cufftDoubleComplex* out, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz) {
    // Calcola la dimensione dell'array complesso in Fourier
    int Nzh = Nz / 2 + 1;
    int size = Nx * Ny * Nzh;

    // Calcola l'indice globale del thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Esce se l'indice è fuori dai limiti
    if (tid >= size) return;

    // Calcolo delle coordinate in Fourier
    int i = tid / (Ny * Nzh);
    int remainder = tid % (Ny * Nzh);
    int j = remainder / Nzh;
    int k = remainder % Nzh;

    // Calcola gli indici del vettore d'onda
    int II = (2 * i < Nx) ? i : Nx - i;
    int JJ = (2 * j < Ny) ? j : Ny - j;
    double k1 = 2.0 * M_PI * II / Lx;
    double k2 = 2.0 * M_PI * JJ / Ly;
    double k3 = 2.0 * M_PI * k / Lz;

    // Fattore di normalizzazione basato sulla norma del vettore d'onda
    double fac = -(k1 * k1 + k2 * k2 + k3 * k3);

    // Aggiorna i valori di `out` in base al fattore
    cufftDoubleComplex val = out[tid];
    if (fabs(fac) < 1e-14) { // Condizione per evitare divisioni per zero
        out[tid].x = 0.0;
        out[tid].y = 0.0;
    } else {
        out[tid].x = val.x / fac;
        out[tid].y = val.y / fac;
    }
}


// Kernel per calcolare i gradienti nello spazio di Fourier.
__global__ void solveGradientFourierKernel(
    cufftDoubleComplex* out, 
    cufftDoubleComplex* outX, cufftDoubleComplex* outY, cufftDoubleComplex* outZ, 
    double Lx, double Ly, double Lz,
    int Nx, int Ny, int Nz, 
    int sign
) 
{
    // Calcola la dimensione dell'array complesso in Fourier
    int Nzh = Nz / 2 + 1;

    // Calcola l'indice globale del thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Esce se l'indice è fuori dai limiti
    if (tid >= Nx * Ny * Nzh) return;

    // Calcolo delle coordinate in Fourier
    int i = tid / (Ny * Nzh);
    int remainder = tid % (Ny * Nzh);
    int j = remainder / Nzh;
    int k = remainder % Nzh;

    // Calcola gli indici del vettore d'onda
    int II = (2 * i < Nx) ? i : i - Nx;
    int JJ = (2 * j < Ny) ? j : j - Ny;
    double k1 = 2.0 * M_PI * II / Lx;
    double k2 = 2.0 * M_PI * JJ / Ly;
    double k3 = 2.0 * M_PI * k / Lz;

    // Recupera il valore dal campo complesso
    cufftDoubleComplex value = out[tid];

    // Calcolo dei gradienti
    outX[tid].x = -value.y * k1 * sign;
    outX[tid].y =  value.x * k1 * sign;

    outY[tid].x = -value.y * k2 * sign;
    outY[tid].y =  value.x * k2 * sign;

    outZ[tid].x = -value.y * k3 * sign;
    outZ[tid].y =  value.x * k3 * sign;
}

#endif
