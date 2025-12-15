#pragma once

#include "parameters.hpp"
#include "structures.hpp"
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#define BLOCK_SIZE 128


namespace depth_level_2
{
  /** calculate the weights given the position of particles 0,0,0 is the left,left, left node */
  void calculateWeights(double weight[][2][2], double xp, double yp, double zp, double qp, int ix, int iy, int iz, simu_grid * grid, double invVOL)
  {
    double xi[2], eta[2], zeta[2];

    xi[1]   = xp - grid->nodeX[ix];
    eta[1]  = yp - grid->nodeY[iy];
    zeta[1] = zp - grid->nodeZ[iz];

    xi[0]   = grid->nodeX[ix+1] - xp;
    eta[0]  = grid->nodeY[iy+1] - yp;
    zeta[0] = grid->nodeZ[iz+1] - zp;

    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
          weight[i][j][k] = xi[i] * eta[j] * zeta[k] * invVOL * qp;
  }

}  // end of depth_level_2 namespace

__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void particles2GridKernel(
    double* d_rhos, double* d_nodeX, double* d_nodeY, double* d_nodeZ,
    double* d_rx, double* d_ry, double* d_rz, double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indices of the cell to which the particle belongs
    int ix = int(d_rx[tid] / dx);
    int iy = int(d_ry[tid] / dy);
    int iz = int(d_rz[tid] / dz);

    // Weight calculation inline
    double xi[2], eta[2], zeta[2];
    double weights[2][2][2];

    // Calculate distances for weight computation
    xi[1]   = d_rx[tid] - d_nodeX[ix];
    eta[1]  = d_ry[tid] - d_nodeY[iy];
    zeta[1] = d_rz[tid] - d_nodeZ[iz];

    xi[0]   = d_nodeX[ix + 1] - d_rx[tid];
    eta[0]  = d_nodeY[iy + 1] - d_ry[tid];
    zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];

    // Calculate weights
    double qp = d_q[tid];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                weights[i][j][k] = xi[i] * eta[j] * zeta[k] * invVOL * qp;

    // Update the grid with particle contributions
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int ixn = (ix + i == nx) ? ix - (nx - 1) - 1 : ix;
                int iyn = (iy + j == ny) ? iy - (ny - 1) - 1 : iy;
                int izn = (iz + k == nz) ? iz - (nz - 1) - 1 : iz;

                if (nx == 1) ix = -i;
                if (ny == 1) iy = -j;
                if (nz == 1) iz = -k;

                int index = is * nx * ny * nz + (ixn + i) * ny * nz + (iyn + j) * nz + (izn + k);
                atomicAdd(&d_rhos[index], weights[i][j][k] * invVOL);
            }
        }
    }
}


/* VERSIONI SEPARATE DELLA PRIVATIZZAZIONE
   DUE KERNEL, UNO PER IL CALCOLO SULLE COPIE
   PRIVATE, E UNO PER MERGIARE I RISULTATI IN d_rhos

__global__ void mergePrivateCopies(
    double* d_rhos,           // The final array to be merged into
    const double* rhos_private, // The array of private copies (size = blocksPerGrid * nCells)
    int nCells,               // Number of cells in the domain
    int blocksPerGrid,         // Number of blocks that wrote into rhos_private
    int is
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nCells) return;

    // Accumulate all private copies into 'sum'
    double sum = 0.0;
    for (int block = 0; block < blocksPerGrid; ++block) {
        int privateIdx = block * nCells + tid;
        sum += rhos_private[privateIdx];
    }

    // Write final result into d_rhos using atomic add
    atomicAdd(&d_rhos[(is * nCells) + tid], sum);
}


__global__ void particles2GridKernelPrivate(
    double* d_rhos, 
    const double* d_nodeX, const double* d_nodeY, const double* d_nodeZ,
    const double* d_rx, const double* d_ry, const double* d_rz, const double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    int is, int np, double* rhos_private) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indices of the cell to which the particle belongs
    int ix = int(d_rx[tid] / dx);
    int iy = int(d_ry[tid] / dy);
    int iz = int(d_rz[tid] / dz);

    // Weight calculation inline
    double xi[2], eta[2], zeta[2];
    double weights[2][2][2];

    // Calculate distances for weight computation
    xi[1]   = d_rx[tid] - d_nodeX[ix];
    eta[1]  = d_ry[tid] - d_nodeY[iy];
    zeta[1] = d_rz[tid] - d_nodeZ[iz];

    xi[0]   = d_nodeX[ix + 1] - d_rx[tid];
    eta[0]  = d_nodeY[iy + 1] - d_ry[tid];
    zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];

    // Calculate weights
    double qp = d_q[tid];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                weights[i][j][k] = xi[i] * eta[j] * zeta[k] * invVOL * qp;

    // Update the private grid with particle contributions
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int ixn = (ix + i == nx) ? ix - (nx - 1) - 1 : ix;
                int iyn = (iy + j == ny) ? iy - (ny - 1) - 1 : iy;
                int izn = (iz + k == nz) ? iz - (nz - 1) - 1 : iz;

                int cellIdx = (ixn + i) * ny * nz + (iyn + j) * nz + (izn + k);
                int privateIdx = blockIdx.x * nx * ny * nz + cellIdx;

                atomicAdd(&rhos_private[privateIdx], weights[i][j][k] * invVOL);
            }
        }
    }
}

*/


// IL VERO KERNEL PRIVATIZZATO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void particles2GridKernelPrivatized(
    double* d_rhos, 
    const double* d_nodeX, const double* d_nodeY, const double* d_nodeZ,
    const double* d_rx, const double* d_ry, const double* d_rz, 
    const double* d_q,
    int nx, int ny, int nz, 
    double dx, double dy, double dz, 
    double invVOL,
    int is,          
    int np,          
    double* rhos_private,
    int nCells, 
    int blocksPerGrid
)
{
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize   = blockDim.x;
    int blockId     = blockIdx.x;
    int blockOffset = blockId * nCells;

    /*
    for (int i = threadIdx.x; i < nCells; i += blockSize) {
        rhos_private[blockOffset + i] = 0.0;
    }
    __syncthreads();
    */

    int particlesPerBlock = blockSize;
    int start = blockId * particlesPerBlock; 
    int p = start + threadIdx.x;
    if (p < np) 
    {
        int ix = int(d_rx[p] / dx);
        int iy = int(d_ry[p] / dy);
        int iz = int(d_rz[p] / dz);

        double xi[2], eta[2], zeta[2];
        double weights[2][2][2];

        xi[1]   = d_rx[p] - d_nodeX[ix];
        eta[1]  = d_ry[p] - d_nodeY[iy];
        zeta[1] = d_rz[p] - d_nodeZ[iz];
        xi[0]   = d_nodeX[ix + 1] - d_rx[p];
        eta[0]  = d_nodeY[iy + 1] - d_ry[p];
        zeta[0] = d_nodeZ[iz + 1] - d_rz[p];

        double qp = d_q[p];

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    weights[i][j][k] = xi[i] * eta[j] * zeta[k] * invVOL * qp;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    int ixn = (ix + i == nx) ? ix - (nx - 1) - 1 : ix;
                    int iyn = (iy + j == ny) ? iy - (ny - 1) - 1 : iy;
                    int izn = (iz + k == nz) ? iz - (nz - 1) - 1 : iz;

                    int cellIdx = (ixn + i) * ny * nz + (iyn + j) * nz + (izn + k);
                    
                    atomicAdd(&rhos_private[blockOffset + cellIdx], weights[i][j][k] * invVOL);
                }
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < nCells; i += blockSize) {
        double val = rhos_private[blockOffset + i];
        if (val != 0.0) {
            atomicAdd(&d_rhos[is * nCells + i], val);
        }
    }
}



__global__ void particles2GridKernelShared(
    double* d_rhos, double* d_nodeX, double* d_nodeY, double* d_nodeZ,
    double* d_rx, double* d_ry, double* d_rz, double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np) {

    extern __shared__ double shared_rhos[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    int blockSizeX = blockDim.x;
    int sharedSizeX = blockSizeX + 2; 
    int sharedSizeY = 3; 
    int sharedSizeZ = 3;

    // Compute the cell indices
    int ix = int(d_rx[tid] / dx);
    int iy = int(d_ry[tid] / dy);
    int iz = int(d_rz[tid] / dz);

    // Compute weights as in the original kernel
    double xi[2], eta[2], zeta[2];
    double weights[2][2][2];
    double qp = d_q[tid];

    xi[1] = d_rx[tid] - d_nodeX[ix];
    eta[1] = d_ry[tid] - d_nodeY[iy];
    zeta[1] = d_rz[tid] - d_nodeZ[iz];

    xi[0] = d_nodeX[ix + 1] - d_rx[tid];
    eta[0] = d_nodeY[iy + 1] - d_ry[tid];
    zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                weights[i][j][k] = xi[i] * eta[j] * zeta[k] * invVOL * qp;

    // Initialize shared memory
    int totalShared = sharedSizeX * sharedSizeY * sharedSizeZ;
    for (int idx = threadIdx.x; idx < totalShared; idx += blockDim.x) {
        shared_rhos[idx] = 0.0;
    }
    __syncthreads();

    // Local indices in shared memory for the "central cell"
    int lx = threadIdx.x + 1; 
    int ly = 1;              
    int lz = 1;              

    // Accumulate to shared memory
    // Apply the second invVOL here to match the original final double invVOL
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int shared_lx = lx + i;
                int shared_ly = ly + j;
                int shared_lz = lz + k;

                int sIdx = shared_lx +
                           shared_ly * sharedSizeX +
                           shared_lz * sharedSizeX * sharedSizeY;

                atomicAdd(&shared_rhos[sIdx], weights[i][j][k] * invVOL);
            }
        }
    }
    __syncthreads();

    // Now write back to global memory EXACTLY as in original kernel
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int ixn = (ix + i == nx) ? ix - (nx - 1) - 1 : ix;
                int iyn = (iy + j == ny) ? iy - (ny - 1) - 1 : iy;
                int izn = (iz + k == nz) ? iz - (nz - 1) - 1 : iz;

                // Replicate the original one-dimensional domain logic
                if (nx == 1) ix = -i;
                if (ny == 1) iy = -j;
                if (nz == 1) iz = -k;

                int shared_lx2 = (threadIdx.x + 1) + i;
                int shared_ly2 = 1 + j;
                int shared_lz2 = 1 + k;
                int sIdx = shared_lx2 +
                           shared_ly2 * sharedSizeX +
                           shared_lz2 * sharedSizeX * sharedSizeY;

                // Only write if inside the domain
                if ( (ixn + i) >= 0 && (ixn + i) < nx &&
                     (iyn + j) >= 0 && (iyn + j) < ny &&
                     (izn + k) >= 0 && (izn + k) < nz ) {

                    int index = is * nx * ny * nz + (ixn + i)*ny*nz + (iyn + j)*nz + (izn + k);
                    double val = shared_rhos[sIdx];
                    // No additional invVOL here, we've already applied invVOL twice in total
                    atomicAdd(&d_rhos[index], val);
                }
            }
        }
    }
}



__global__ void particles2GridKernelPrivatization1(
    double* d_rhos, double* d_nodeX, double* d_nodeY, double* d_nodeZ,
    double* d_rx, double* d_ry, double* d_rz, double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np) {

    extern __shared__ double s_rhos[]; // Memoria condivisa per ogni blocco

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    int localIdx = threadIdx.x; // Indice del thread all'interno del blocco

    // Dimensione della griglia per questa specie
    int gridSize = nx * ny * nz;

    // Inizializza la memoria condivisa
    for (int i = localIdx; i < gridSize; i += blockDim.x) {
        s_rhos[i] = 0.0;
    }
    __syncthreads();

    // Indici della cella a cui appartiene la particella
    int ix = int(d_rx[tid] / dx);
    int iy = int(d_ry[tid] / dy);
    int iz = int(d_rz[tid] / dz);

    // Calcolo pesi inline
    double xi[2], eta[2], zeta[2];
    double weights[2][2][2];

    xi[1]   = d_rx[tid] - d_nodeX[ix];
    eta[1]  = d_ry[tid] - d_nodeY[iy];
    zeta[1] = d_rz[tid] - d_nodeZ[iz];

    xi[0]   = d_nodeX[ix + 1] - d_rx[tid];
    eta[0]  = d_nodeY[iy + 1] - d_ry[tid];
    zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];

    double qp = d_q[tid];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                weights[i][j][k] = xi[i] * eta[j] * zeta[k] * invVOL * qp;

    // Aggiornamento della memoria condivisa
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                int ixn = (ix + i == nx) ? ix - (nx - 1) - 1 : ix;
                int iyn = (iy + j == ny) ? iy - (ny - 1) - 1 : iy;
                int izn = (iz + k == nz) ? iz - (nz - 1) - 1 : iz;

                if (nx == 1) ix = -i;
                if (ny == 1) iy = -j;
                if (nz == 1) iz = -k;

                int index = (ixn + i) * ny * nz + (iyn + j) * nz + (izn + k);
                atomicAdd(&s_rhos[index], weights[i][j][k]);
            }
        }
    }
    __syncthreads();

    // Scrive i dati dalla memoria condivisa alla memoria globale
    for (int i = localIdx; i < gridSize; i += blockDim.x) {
        atomicAdd(&d_rhos[is * gridSize + i], s_rhos[i]);
    }
}




/*
La privatization consiste nell'utilizzare variabili locali per ogni thread, 
come weights_local e index_local, anziché condividere variabili tra i thread. 
Questo metodo riduce la contesa sulle operazioni atomiche in memoria globale, 
poiché ogni thread lavora sulle proprie copie private dei dati. Di conseguenza, 
si ottimizza l'accesso alla memoria e si migliora la parallelizzazione, aumentando 
le prestazioni rispetto alla versione normale che utilizza variabili condivise.
*/
__global__ void particles2GridKernelPrivatization(
    double* d_rhos, double* d_nodeX, double* d_nodeY, double* d_nodeZ,
    double* d_rx, double* d_ry, double* d_rz, double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indici della cella a cui appartiene la particella
    int ix = int(d_rx[tid] / dx);
    int iy = int(d_ry[tid] / dy);
    int iz = int(d_rz[tid] / dz);

    // Variabili private per i pesi e gli indici
    double weights_local[8];
    int index_local[8];

    // Calcolo delle distanze per il calcolo dei pesi
    double xi[2], eta[2], zeta[2];
    xi[1]   = d_rx[tid] - d_nodeX[ix];
    eta[1]  = d_ry[tid] - d_nodeY[iy];
    zeta[1] = d_rz[tid] - d_nodeZ[iz];

    xi[0]   = d_nodeX[ix + 1] - d_rx[tid];
    eta[0]  = d_nodeY[iy + 1] - d_ry[tid];
    zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];

    // Calcolo dei pesi e degli indici locali
    double qp = d_q[tid];
    int idx = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                weights_local[idx] = xi[i] * eta[j] * zeta[k] * invVOL * qp * invVOL;
                int ixn = (ix + i) % nx;
                int iyn = (iy + j) % ny;
                int izn = (iz + k) % nz;
                index_local[idx] = is * nx * ny * nz + ixn * ny * nz + iyn * nz + izn;
                idx++;
            }

    // Aggiornamento della griglia usando variabili private
    // Il ciclo va da 0 a 8 perché ci sono 2x2x2 = 8 vertici della cella che devono essere aggiornati
    for (int i = 0; i < 8; i++) {
        atomicAdd(&d_rhos[index_local[i]], weights_local[i]);
    }
}


// Versione con Coarsening
/*
Ogni thread processa multiple particelle (PARTICLES_PER_THREAD)
Riduce il numero totale di thread necessari
Migliora l'utilizzo delle risorse quando ci sono molte particelle
Il numero di thread diventa: np / PARTICLES_PER_THREAD (arrotondato per eccesso)*/
__global__ void particles2GridKernelCoarsening(
    double* d_rhos, double* d_nodeX, double* d_nodeY, double* d_nodeZ,
    double* d_rx, double* d_ry, double* d_rz, double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np, int PARTICLES_PER_THREAD) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int particleStart = tid * PARTICLES_PER_THREAD;
    int particleEnd = min(particleStart + PARTICLES_PER_THREAD, np);

    // Processo multiple particelle per thread
    for (int p = particleStart; p < particleEnd; p++) {
        // Indici della cella per questa particella
        int ix = int(d_rx[p] / dx);
        int iy = int(d_ry[p] / dy);
        int iz = int(d_rz[p] / dz);

        // Calcolo delle distanze
        double xi[2], eta[2], zeta[2];
        xi[1]   = d_rx[p] - d_nodeX[ix];
        eta[1]  = d_ry[p] - d_nodeY[iy];
        zeta[1] = d_rz[p] - d_nodeZ[iz];

        xi[0]   = d_nodeX[ix + 1] - d_rx[p];
        eta[0]  = d_nodeY[iy + 1] - d_ry[p];
        zeta[0] = d_nodeZ[iz + 1] - d_rz[p];

        // Calcolo dei pesi e aggiornamento della griglia
        double qp = d_q[p];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    double weight = xi[i] * eta[j] * zeta[k] * invVOL * qp * invVOL;
                    int ixn = (ix + i) % nx;
                    int iyn = (iy + j) % ny;
                    int izn = (iz + k) % nz;
                    int index = is * nx * ny * nz + ixn * ny * nz + iyn * nz + izn;
                    atomicAdd(&d_rhos[index], weight);
                }
    }
}

/*
__global__ void particles2GridKernelShared(
    double* d_rhos, double* d_nodeX, double* d_nodeY, double* d_nodeZ,
    double* d_rx, double* d_ry, double* d_rz, double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    const int is, int np) {
    
    // Blocco di shared memory per accumulo locale
    extern __shared__ double s_rhos[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x; // indice locale nel blocco
    
    // Inizializza shared memory
    s_rhos[lid] = 0.0;
    __syncthreads();
    
    if (tid < np) {
        // Calcolo indici della cella
        int ix = int(d_rx[tid] / dx);
        int iy = int(d_ry[tid] / dy);
        int iz = int(d_rz[tid] / dz);
        
        // Calcolo pesi
        double xi[2], eta[2], zeta[2];
        double weights[2][2][2];
        
        xi[1] = d_rx[tid] - d_nodeX[ix];
        eta[1] = d_ry[tid] - d_nodeY[iy];
        zeta[1] = d_rz[tid] - d_nodeZ[iz];
        
        xi[0] = d_nodeX[ix + 1] - d_rx[tid];
        eta[0] = d_nodeY[iy + 1] - d_ry[tid];
        zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];
        
        // Calcola i pesi
        double qp = d_q[tid];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    weights[i][j][k] = xi[i] * eta[j] * zeta[k] * invVOL * qp;
        
        // Accumulo in shared memory
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    int ixn = (ix + i == nx) ? ix - (nx - 1) - 1 : ix;
                    int iyn = (iy + j == ny) ? iy - (ny - 1) - 1 : iy;
                    int izn = (iz + k == nz) ? iz - (nz - 1) - 1 : iz;
                    
                    if (nx == 1) ix = -i;
                    if (ny == 1) iy = -j;
                    if (nz == 1) iz = -k;
                    
                    // Calcola l'indice lineare per la griglia
                    int grid_idx = is * nx * ny * nz + (ixn + i) * ny * nz + (iyn + j) * nz + (izn + k);
                    int shared_idx = ((ixn + i) * ny * nz + (iyn + j) * nz + (izn + k)) % blockDim.x;
                    
                    atomicAdd(&s_rhos[shared_idx], weights[i][j][k] * invVOL);
                }
            }
        }
    }
    
    // Sincronizza prima della scrittura finale
    __syncthreads();
    
    // Scrivi i risultati in memoria globale
    if (lid < blockDim.x) {
        int grid_idx = blockIdx.x * blockDim.x + lid;
        if (grid_idx < nx * ny * nz) {
            atomicAdd(&d_rhos[is * nx * ny * nz + grid_idx], s_rhos[lid]);
        }
    }
}*/


/**
 *
 * Questo kernel implementa una variante ottimizzata del calcolo della distribuzione
 * di carica delle particelle sulla griglia utilizzando l'approccio di "aggregation".
 *
 * L'ottimizzazione "aggregation" cerca di ridurre il numero di operazioni atomiche
 * sulla memoria globale. Invece di aggiornare direttamente il valore della densità
 * di carica `d_rhos` per ogni contributo di una particella, il kernel accumula i 
 * contributi in una variabile locale (`accumulator`) finché non cambia l'indice della
 * cella. Quando l'indice cambia, il contributo accumulato viene scritto su `d_rhos`
 * utilizzando un'operazione atomica. Alla fine del loop, eventuali contributi 
 * rimanenti vengono anch'essi aggiunti.
 *
 * Questo approccio riduce il numero di chiamate ad `atomicAdd`, migliorando le 
 * prestazioni soprattutto quando più particelle contribuiscono alle stesse celle.
 */

__global__ void particles2GridKernelAggregation(
    double* d_rhos, const double* d_nodeX, const double* d_nodeY, const double* d_nodeZ,
    const double* d_rx, const double* d_ry, const double* d_rz, const double* d_q,
    int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
    int is, int np)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indices of the cell to which the particle belongs
    int ix = int(d_rx[tid] / dx);
    int iy = int(d_ry[tid] / dy);
    int iz = int(d_rz[tid] / dz);

    // Calculate distances for weight computation
    double xi[2], eta[2], zeta[2];
    xi[1]   = d_rx[tid] - d_nodeX[ix];
    eta[1]  = d_ry[tid] - d_nodeY[iy];
    zeta[1] = d_rz[tid] - d_nodeZ[iz];

    xi[0]   = d_nodeX[ix + 1] - d_rx[tid];
    eta[0]  = d_nodeY[iy + 1] - d_ry[tid];
    zeta[0] = d_nodeZ[iz + 1] - d_rz[tid];

    double qp = d_q[tid];

    // Variables for aggregation
    double accumulator = 0.0;
    int prevIndex = -1;

    // Compute contributions for the 2x2x2 vertices
    // We do NOT restore ix, iy, iz each iteration; we follow the original kernel’s logic exactly.
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                // Compute temporary cell indices as in the original kernel
                int ixn = (ix + i == nx) ? ix - (nx - 1) - 1 : ix;
                int iyn = (iy + j == ny) ? iy - (ny - 1) - 1 : iy;
                int izn = (iz + k == nz) ? iz - (nz - 1) - 1 : iz;

                if (nx == 1) ix = -i;
                if (ny == 1) iy = -j;
                if (nz == 1) iz = -k;

                int index = is * nx * ny * nz + (ixn + i) * ny * nz + (iyn + j) * nz + (izn + k);

                // Compute weight exactly as original kernel
                double weight = xi[i] * eta[j] * zeta[k] * invVOL * qp;

                // Original kernel adds weight * invVOL, so we do the same
                double final_val = weight * invVOL;

                // Aggregation logic
                if (index == prevIndex) {
                    accumulator += final_val;
                } else {
                    // Commit previous accumulation if different index
                    if (prevIndex != -1 && accumulator != 0.0) {
                        atomicAdd(&d_rhos[prevIndex], accumulator);
                    }
                    accumulator = final_val;
                    prevIndex = index;
                }
            }
        }
    }

    // Commit any leftover accumulation
    if (prevIndex != -1 && accumulator != 0.0) {
        atomicAdd(&d_rhos[prevIndex], accumulator);
    }
}




__global__ void updateParticlePositionKernel(double* rx, double* ry, double* rz,
                                             double* vx, double* vy, double* vz,
                                             const int np, double Lx, double Ly, double Lz, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Indice globale del thread
    if (idx < np) // Assicura di non accedere oltre il numero di particelle
    {
        // Update della posizione
        rx[idx] += vx[idx] * dt;
        ry[idx] += vy[idx] * dt;
        rz[idx] += vz[idx] * dt;

        // Periodic boundary conditions
        if (rx[idx] >= Lx) rx[idx] = fmod(rx[idx], Lx);
        if (rx[idx] < 0)   rx[idx] = fmod(rx[idx] + Lx, Lx);

        if (ry[idx] >= Ly) ry[idx] = fmod(ry[idx], Ly);
        if (ry[idx] < 0)   ry[idx] = fmod(ry[idx] + Ly, Ly);

        if (rz[idx] >= Lz) rz[idx] = fmod(rz[idx], Lz);
        if (rz[idx] < 0)   rz[idx] = fmod(rz[idx] + Lz, Lz);

        // Aggiustamento per valori al limite
        if (rx[idx] == Lx) rx[idx] = 0;
        if (ry[idx] == Ly) ry[idx] = 0;
        if (rz[idx] == Lz) rz[idx] = 0;
    }
}


/** calculate new particle position */
void updateParticlePosition(simu_particles** part, const unsigned int is, int np[], double Lx, double Ly, double Lz, double dt)
{
    for (int i = 0; i < np[is]; i++)
    {
        // update particle position
        part[is]->rx[i] += part[is]->vx[i] * dt;
        part[is]->ry[i] += part[is]->vy[i] * dt;
        part[is]->rz[i] += part[is]->vz[i] * dt;

        // periodic boundary conditions
        if (part[is]->rx[i] >= Lx) part[is]->rx[i] = (part[is]->rx[i]/Lx - int(     part[is]->rx[i]/Lx)) * Lx;
        if (part[is]->rx[i] <  0)  part[is]->rx[i] = (part[is]->rx[i]/Lx + int(fabs(part[is]->rx[i]/Lx)) + 1) * Lx;
        if (part[is]->ry[i] >= Ly) part[is]->ry[i] = (part[is]->ry[i]/Ly - int(     part[is]->ry[i]/Ly)) * Ly;
        if (part[is]->ry[i] <  0)  part[is]->ry[i] = (part[is]->ry[i]/Ly + int(fabs(part[is]->ry[i]/Ly)) + 1) * Ly;
        if (part[is]->rz[i] >= Lz) part[is]->rz[i] = (part[is]->rz[i]/Lz - int(     part[is]->rz[i]/Lz)) * Lz;
        if (part[is]->rz[i] <  0)  part[is]->rz[i] = (part[is]->rz[i]/Lz + int(fabs(part[is]->rz[i]/Lz)) + 1) * Lz;

        if (part[is]->rx[i] == Lx) part[is]->rx[i] = 0;
        if (part[is]->ry[i] == Ly) part[is]->ry[i] = 0;
        if (part[is]->rz[i] == Lz) part[is]->rz[i] = 0;
    }
}


/** Interpolation Particle --> Grid */
void particles2Grid(simu_fields * fields, 
                    simu_grid * grid, int nx, int ny, int nz, double dx, double dy, double dz, double invVOL,
                    simu_particles** part, const int is, int np[]) 
{
  // indices of the cell to which the particle belongs
  int ix, iy, iz;    

  // Indices to account for periodic boundary conditions
  int ixn, iyn, izn; 
  
  // matrix of cell vertex weights
  double weights[2][2][2];

  for (int i = 0; i < np[is]; i++)
  {
    ix = int(part[is]->rx[i] / dx);
    iy = int(part[is]->ry[i] / dy);
    iz = int(part[is]->rz[i] / dz);
   
    depth_level_2::calculateWeights(weights, part[is]->rx[i], part[is]->ry[i], part[is]->rz[i], part[is]->q[i], ix, iy, iz, grid, invVOL); // <<< convert to a cuda device function >>> 
    
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) 
        {
          ((ix + i) == nx) ? ixn = ix - (nx -1) -1 : ixn = ix;
          ((iy + j) == ny) ? iyn = iy - (ny -1) -1 : iyn = iy;
          ((iz + k) == nz) ? izn = iz - (nz -1) -1 : izn = iz;

          if (nx == 1) ix = -i;
          if (ny == 1) iy = -j;
          if (nz == 1) iz = -k;
          
          int index = is*nx*ny*nz + (ixn + i)*ny*nz + (iyn + j)*nz + (izn + k);
          fields->rhos[index] += weights[i][j][k] * invVOL;  // <<< PAY ATTENTION: possible race condition >>>
        }
  }
}

__global__ void updateParticleVelocityKernel(double* rx, double* ry, double* rz,
                                  double* vx, double* vy, double* vz,
                                  double* Exn, double* Eyn, double* Ezn,
                                  double* nodeX, double* nodeY, double* nodeZ,
                                  int nx, int ny, int nz, 
                                  double dx, double dy, double dz, double invVOL,
                                  double qom, double dt, int np) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= np) return;

    // Indici della cella
    int ix = int(rx[i] / dx);
    int iy = int(ry[i] / dy);
    int iz = int(rz[i] / dz);

    // Peso dei vertici della cella
    double weights[2][2][2] = {0};

    // Calcolo dei pesi
    double xi[2], eta[2], zeta[2];
    xi[1]   = rx[i] - nodeX[ix];
    eta[1]  = ry[i] - nodeY[iy];
    zeta[1] = rz[i] - nodeZ[iz];
    xi[0]   = nodeX[ix + 1] - rx[i];
    eta[0]  = nodeY[iy + 1] - ry[i];
    zeta[0] = nodeZ[iz + 1] - rz[i];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                weights[ii][jj][kk] = fabs(xi[ii] * eta[jj] * zeta[kk]) * invVOL;

    // Campo elettrico interpolato
    double Ep[3] = {0.0, 0.0, 0.0};
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++) {
                int ixn = (ix + ii) % nx;
                int iyn = (iy + jj) % ny;
                int izn = (iz + kk) % nz;

                int index = ixn * ny * nz + iyn * nz + izn;
                Ep[0] += weights[ii][jj][kk] * Exn[index];
                Ep[1] += weights[ii][jj][kk] * Eyn[index];
                Ep[2] += weights[ii][jj][kk] * Ezn[index];
            }

    // Aggiorna la velocità
    vx[i] += qom * Ep[0] * dt;
    vy[i] += qom * Ep[1] * dt;
    vz[i] += qom * Ep[2] * dt;
}

// /* Interpolation Grid --> Particle */
void updateParticleVelocity(simu_particles** part, const unsigned int is, int np[], double qom[], double dt,
                            simu_fields * fields, 
                            simu_grid * grid, int nx, int ny, int nz, double dx, double dy, double dz, double invVOL)
{
  // indices of the cell to which the particle belongs
  int ix, iy, iz;

  // Indices to account for periodic boundary conditions
  int ixn, iyn, izn; 

  // matrix of cell vertex weights
  double weights[2][2][2]; 
    
  for (int i = 0; i < np[is]; i++)
  {
    double Ep[3] = {0.0, 0.0, 0.0};
    
    ix = int(part[is]->rx[i] / dx);
    iy = int(part[is]->ry[i] / dy);
    iz = int(part[is]->rz[i] / dz);
    
    depth_level_2::calculateWeights(weights, part[is]->rx[i], part[is]->ry[i], part[is]->rz[i], 1.0, ix, iy, iz, grid, invVOL);
    
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) 
        {
          ((ix + i) == nx) ? ixn = ix - (nx -1) -1 : ixn = ix;
          ((iy + j) == ny) ? iyn = iy - (ny -1) -1 : iyn = iy;
          ((iz + k) == nz) ? izn = iz - (nz -1) -1 : izn = iz;  
          
          if (nx == 1) ix = -i;
          if (ny == 1) iy = -j;
          if (nz == 1) iz = -k;
          
          int index = (ixn + i)*ny*nz + (iyn + j)*nz + (izn + k);
          Ep[0] += fabs(weights[i][j][k]) * fields->Exn[index];
          Ep[1] += fabs(weights[i][j][k]) * fields->Eyn[index];
          Ep[2] += fabs(weights[i][j][k]) * fields->Ezn[index];
        }
    
      // update particle velocity
      part[is]->vx[i] += qom[is] * Ep[0] * dt;
      part[is]->vy[i] += qom[is] * Ep[1] * dt;
      part[is]->vz[i] += qom[is] * Ep[2] * dt;
  }
}
