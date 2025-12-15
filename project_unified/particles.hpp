#pragma once

#include "parameters.hpp"
#include "structures.hpp"
#include <cuda_runtime.h>
#include <device_atomic_functions.h>


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

//VERSIONI KERNEL
__global__ void updateParticlePositionKernel(
    double* rx, double* ry, double* rz, // vettore posizione della microparticella
    double* vx, double* vy, double* vz, // vettore velocità della microparticella
    const int np,                       // numero totale delle microparticelle, data una specie (la chiamata al kernel passa np[is] dove is è la specie)
    double Lx, double Ly, double Lz,    // limiti oltre i quali le microparticelle non posso andare, lungo x y e z
    double dt                           // passo temporale (intervallo di tempo tra uno step e l'altro(?))
)
{
    // qui il dominio è sempre unidimensionale, quindi ci calcoliamo
    // solo lungo x l'indice della microparticella.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < np)
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

// IL VERO KERNEL PRIVATIZZATO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void particles2GridKernelPrivatized(
    simu_particles **part, simu_grid *grid, simu_fields *fields,
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
        int ix = int(part[is]->rx[p] / dx);
        int iy = int(part[is]->ry[p] / dy);
        int iz = int(part[is]->rz[p] / dz);

        double xi[2], eta[2], zeta[2];
        double weights[2][2][2];

        xi[1]   = part[is]->rx[p] - grid->nodeX[ix];
        eta[1]  = part[is]->ry[p] - grid->nodeY[iy];
        zeta[1] = part[is]->rz[p] - grid->nodeZ[iz];
        xi[0]   = grid->nodeX[ix + 1] - part[is]->rx[p];
        eta[0]  = grid->nodeY[iy + 1] - part[is]->ry[p];
        zeta[0] = grid->nodeZ[iz + 1] - part[is]->rz[p];

        double qp = part[is]->q[p];

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
            atomicAdd(&fields->rhos[is * nCells + i], val);
        }
    }
}




__global__ void particles2GridKernel(
    simu_fields* fields,                // struct che rappresenta il campo elettromagnetico
    simu_grid* grid,                    // struct che rappresenta il dominio 3D per il PIC
    int nx, int ny, int nz,             // quante MACROparticelle abbiamo lungo x y e z
    double dx, double dy, double dz,    // quanto sono lunghe le MACROparticelle lungo x y e z
    double invVOL,                      // Fattore di normalizzazione per il volume della cella.
    simu_particles** part,              // array di struct che rappresentano le microparticelle
    const int is,                       // specie che si vuole computare al momento
    int np                              // numero totale delle microparticelle, data una specie (la chiamata al kernel passa np[is] dove is è la specie)
) 
{
    // qui il dominio è sempre unidimensionale, quindi ci calcoliamo
    // solo lungo x l'indice della microparticella.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indices of the cell to which the particle belongs
    int ix = int(part[is]->rx[tid] / dx);
    int iy = int(part[is]->ry[tid] / dy);
    int iz = int(part[is]->rz[tid] / dz);

    // Weight calculation inline
    double xi[2], eta[2], zeta[2];
    double weights[2][2][2];

    // Calculate distances for weight computation
    xi[1]   = part[is]->rx[tid] - grid->nodeX[ix];
    eta[1]  = part[is]->ry[tid] - grid->nodeY[iy];
    zeta[1] = part[is]->rz[tid] - grid->nodeZ[iz];

    xi[0]   = grid->nodeX[ix+1] - part[is]->rx[tid];
    eta[0]  = grid->nodeY[iy+1] - part[is]->ry[tid];
    zeta[0] = grid->nodeZ[iz+1] - part[is]->rz[tid];

    // Calculate weights
    double qp = part[is]->q[tid];
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

                int index = is*nx*ny*nz + (ixn + i)*ny*nz + (iyn + j)*nz + (izn + k);
                atomicAdd(&fields->rhos[index], weights[i][j][k] * invVOL);
            }
        }
    }
}

__global__ void particles2GridKernelPrivatization(
    simu_fields* fields, simu_grid* grid,
    int nx, int ny, int nz, 
    double dx, double dy, double dz, double invVOL,
    simu_particles** part, int is, int np) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Calcolo degli indici della cella
    double xp = part[is]->rx[tid];
    double yp = part[is]->ry[tid];
    double zp = part[is]->rz[tid];

    // Gestione delle condizioni periodiche
    if (xp < 0) xp += nx * dx;
    if (yp < 0) yp += ny * dy;
    if (zp < 0) zp += nz * dz;
    if (xp > nx * dx) xp -= nx * dx;
    if (yp > ny * dy) yp -= ny * dy;
    if (zp > nz * dz) zp -= nz * dz;

    // Calcolo degli indici corretti
    int ix = (int)(xp / dx);
    int iy = (int)(yp / dy);
    int iz = (int)(zp / dz);

    // Assicuriamoci che gli indici siano validi
    ix = (ix < 0) ? 0 : ((ix >= nx) ? nx-1 : ix);
    iy = (iy < 0) ? 0 : ((iy >= ny) ? ny-1 : iy);
    iz = (iz < 0) ? 0 : ((iz >= nz) ? nz-1 : iz);

    // Variabili private per i pesi e gli indici
    double weights_local[8];
    int index_local[8];

    // Calcolo delle distanze normalizzate
    double xi[2], eta[2], zeta[2];
    
    // Coordinate normalizzate della particella all'interno della cella
    double xi_p = (xp - grid->nodeX[ix]) / dx;
    double eta_p = (yp - grid->nodeY[iy]) / dy;
    double zeta_p = (zp - grid->nodeZ[iz]) / dz;

    // Calcolo dei pesi nelle direzioni x, y, z
    xi[0]   = 1.0 - xi_p;
    xi[1]   = xi_p;
    eta[0]  = 1.0 - eta_p;
    eta[1]  = eta_p;
    zeta[0] = 1.0 - zeta_p;
    zeta[1] = zeta_p;

    // Calcolo dei pesi e degli indici locali
    double qp = part[is]->q[tid];
    int idx = 0;
    for (int i = 0; i < 2; i++) {
        int ixn = (ix + i) % nx;
        for (int j = 0; j < 2; j++) {
            int iyn = (iy + j) % ny;
            for (int k = 0; k < 2; k++) {
                int izn = (iz + k) % nz;
                weights_local[idx] = qp * xi[i] * eta[j] * zeta[k] * invVOL;
                index_local[idx] = is * (nx * ny * nz) + (ixn * ny + iyn) * nz + izn;
                idx++;
            }
        }
    }

    // Aggiornamento della griglia usando variabili private
    for (int i = 0; i < 8; i++) {
        atomicAdd(&fields->rhos[index_local[i]], weights_local[i]);
    }
}

__global__ void particles2GridKernelCoarsening(
    simu_fields* fields,                // struct che rappresenta il campo elettromagnetico
    simu_grid* grid,                    // struct che rappresenta il dominio 3D per il PIC
    int nx, int ny, int nz,             // quante MACROparticelle abbiamo lungo x y e z
    double dx, double dy, double dz,    // quanto sono lunghe le MACROparticelle lungo x y e z
    double invVOL,                      // Fattore di normalizzazione per il volume della cella
    simu_particles** part,              // array di struct che rappresentano le microparticelle
    const int is,                       // specie che si vuole computare al momento
    int np,                             // numero totale delle microparticelle
    int PARTICLES_PER_THREAD            // numero di particelle gestite da ogni thread
) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int particleStart = tid * PARTICLES_PER_THREAD;
    int particleEnd = min(particleStart + PARTICLES_PER_THREAD, np);

    // Processo multiple particelle per thread
    for (int p = particleStart; p < particleEnd; p++) {
        // Indici della cella per questa particella
        int ix = int(part[is]->rx[p] / dx);
        int iy = int(part[is]->ry[p] / dy);
        int iz = int(part[is]->rz[p] / dz);

        // Weight calculation inline
        double xi[2], eta[2], zeta[2];

        // Calculate distances for weight computation
        xi[1]   = part[is]->rx[p] - grid->nodeX[ix];
        eta[1]  = part[is]->ry[p] - grid->nodeY[iy];
        zeta[1] = part[is]->rz[p] - grid->nodeZ[iz];

        xi[0]   = grid->nodeX[ix+1] - part[is]->rx[p];
        eta[0]  = grid->nodeY[iy+1] - part[is]->ry[p];
        zeta[0] = grid->nodeZ[iz+1] - part[is]->rz[p];

        double qp = part[is]->q[p];

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

                    int index = is*nx*ny*nz + (ixn + i)*ny*nz + (iyn + j)*nz + (izn + k);
                    
                    // Calculate weight
                    double weight = xi[i] * eta[j] * zeta[k] * invVOL * qp * invVOL;
                    atomicAdd(&fields->rhos[index], weight);
                }
            }
        }
    }
}

__global__ void particles2GridKernelAggregation(
    simu_fields* fields,                // struct che rappresenta il campo elettromagnetico
    simu_grid* grid,                    // struct che rappresenta il dominio 3D per il PIC
    int nx, int ny, int nz,             // quante MACROparticelle abbiamo lungo x y e z
    double dx, double dy, double dz,    // quanto sono lunghe le MACROparticelle lungo x y e z
    double invVOL,                      // Fattore di normalizzazione per il volume della cella.
    simu_particles** part,              // array di struct che rappresentano le microparticelle
    const int is,                       // specie che si vuole computare al momento
    int np                              // numero totale delle microparticelle
) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= np) return;

    // Indices of the cell to which the particle belongs
    int ix = int(part[is]->rx[tid] / dx);
    int iy = int(part[is]->ry[tid] / dy);
    int iz = int(part[is]->rz[tid] / dz);

    // Weight calculation inline
    double xi[2], eta[2], zeta[2];

    // Calculate distances for weight computation
    xi[1]   = part[is]->rx[tid] - grid->nodeX[ix];
    eta[1]  = part[is]->ry[tid] - grid->nodeY[iy];
    zeta[1] = part[is]->rz[tid] - grid->nodeZ[iz];

    xi[0]   = grid->nodeX[ix+1] - part[is]->rx[tid];
    eta[0]  = grid->nodeY[iy+1] - part[is]->ry[tid];
    zeta[0] = grid->nodeZ[iz+1] - part[is]->rz[tid];

    double qp = part[is]->q[tid];

    // Variables for aggregation
    double accumulator = 0.0;
    int prevIndex = -1;

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

                int index = is*nx*ny*nz + (ixn + i)*ny*nz + (iyn + j)*nz + (izn + k);
                
                // Calculate weight
                double weight = xi[i] * eta[j] * zeta[k] * invVOL * qp;
                double final_val = weight * invVOL;

                // Aggregation logic
                if (index == prevIndex) {
                    accumulator += final_val;
                } else {
                    // Commit previous accumulation if different index
                    if (prevIndex != -1 && accumulator != 0.0) {
                        atomicAdd(&fields->rhos[prevIndex], accumulator);
                    }
                    accumulator = final_val;
                    prevIndex = index;
                }
            }
        }
    }

    // Commit any leftover accumulation
    if (prevIndex != -1 && accumulator != 0.0) {
        atomicAdd(&fields->rhos[prevIndex], accumulator);
    }
}

// Kernel per aggiornare la velocità delle particelle in base al campo elettrico interpolato.
// Questo kernel viene eseguito su un thread per particella.
__global__ void updateParticleVelocityKernel(
    double* rx, double* ry, double* rz,          // Posizioni delle particelle (array separati per x, y, z).
    double* vx, double* vy, double* vz,          // Velocità delle particelle (array separati per x, y, z).
    double* Exn, double* Eyn, double* Ezn,       // Componenti del campo elettrico sulla griglia (x, y, z).
    double* nodeX, double* nodeY, double* nodeZ, // Posizioni dei nodi della griglia in ogni direzione.
    int nx, int ny, int nz,                      // Numero di nodi della griglia in x, y, z.
    double dx, double dy, double dz,             // Distanza tra i nodi della griglia in x, y, z.
    double invVOL,                               // Fattore di normalizzazione per il volume della cella.
    double qom,                                  // Rapporto carica/massa della particella.
    double dt,                                   // Intervallo di tempo per l'aggiornamento.
    int np                                       // Numero totale di particelle.
) 
{
    // Calcola l'indice globale della particella gestita da questo thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Controlla che l'indice della particella sia valido
    if (i >= np) return;

    // Identifica la cella della griglia in cui si trova la particella
    int ix = int(rx[i] / dx);  // Indice della cella lungo x
    int iy = int(ry[i] / dy);  // Indice della cella lungo y
    int iz = int(rz[i] / dz);  // Indice della cella lungo z

    // Matrice per i pesi dei nodi della cella
    double weights[2][2][2] = {0};

    // Calcolo delle distanze relative tra la particella e i nodi della cella
    double xi[2], eta[2], zeta[2];
    xi[1]   = rx[i] - nodeX[ix];           // Distanza della particella dal nodo inferiore lungo x
    eta[1]  = ry[i] - nodeY[iy];           // Distanza della particella dal nodo inferiore lungo y
    zeta[1] = rz[i] - nodeZ[iz];           // Distanza della particella dal nodo inferiore lungo z
    xi[0]   = nodeX[ix + 1] - rx[i];       // Distanza della particella dal nodo superiore lungo x
    eta[0]  = nodeY[iy + 1] - ry[i];       // Distanza della particella dal nodo superiore lungo y
    zeta[0] = nodeZ[iz + 1] - rz[i];       // Distanza della particella dal nodo superiore lungo z

    // Calcolo dei pesi dei nodi in base alle distanze
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                weights[ii][jj][kk] = fabs(xi[ii] * eta[jj] * zeta[kk]) * invVOL;

    // Interpolazione del campo elettrico nei nodi vicini
    double Ep[3] = {0.0, 0.0, 0.0}; // Componente interpolata del campo elettrico in x, y, z
    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            for (int kk = 0; kk < 2; kk++) {
                // Calcola gli indici della cella rispettando le condizioni periodiche
                int ixn = (ix + ii) % nx;
                int iyn = (iy + jj) % ny;
                int izn = (iz + kk) % nz;

                // Calcola l'indice lineare del nodo corrente nella griglia
                int index = ixn * ny * nz + iyn * nz + izn;

                // Accumula i contributi ponderati del campo elettrico
                Ep[0] += weights[ii][jj][kk] * Exn[index];
                Ep[1] += weights[ii][jj][kk] * Eyn[index];
                Ep[2] += weights[ii][jj][kk] * Ezn[index];
            }
        }
    }

    // Aggiorna le velocità della particella utilizzando il campo elettrico interpolato
    vx[i] += qom * Ep[0] * dt; // Componente x della velocità
    vy[i] += qom * Ep[1] * dt; // Componente y della velocità
    vz[i] += qom * Ep[2] * dt; // Componente z della velocità
}
