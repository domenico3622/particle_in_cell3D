#ifndef INIT_HPP
#define INIT_HPP

#include <math.h>
#include "parameters.hpp"
#include "structures.hpp"

// FUNZIONI DI BASE NON TOCCATE ------------------------------------------------------------------------------------------
/*! Maxellian random velocity and uniform spatial distribution */
void maxwellian(simu_particles** part, int is, int npc[], int npcx[], int npcy[], int npcz[], double qom[], 
                                       double u0[], double v0[], double w0[],
                                       double uth0[], double vth0[], double wth0[],
                simu_grid * grid, int nx, int ny, int nz, double dx, double dy, double dz,
                simu_fields * EMf)
{
  double harvest;
  double prob, theta;

  int counter = 0;
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

void initPartTwostreams(simu_particles** part, int is, int npc[], int npcx[], int npcy[], int npcz[], double qom[], 
                                               double u0[], double v0[], double w0[], 
                                               double uth0[], double vth0[], double wth0[],
                        simu_grid * grid, int nx, int ny, int nz, double dx, double dy, double dz,
                        simu_fields * EMf)
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
  for (int i = 0; i < nx * ny * nz; i++)
  {
    for (int is = 0; is < ns; is++)
      EMf->rhos[is * nx * ny * nz + i] = (qom[is]/fabs(qom[is])) * (1 / (4 * M_PI));

    EMf->Bxn[i] = B0x;
    EMf->Byn[i] = B0y;
    EMf->Bzn[i] = B0z;
  }
 
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
// ---------------------------------------------------------------------------------------------------------------------

// FUNZIONI CUDA ---------------------------------------------------------------------------------------------------------
__global__ void maxwellianKernelStandard(double* d_rx, double* d_ry, double* d_rz, double* d_q, 
                                        double* d_vx, double* d_vy, double* d_vz, int* d_ID,
                                        double* d_nodeX, double* d_nodeY, double* d_nodeZ, 
                                        double* d_rhos,
                                        int npcx, int npcy, int npcz,
                                        int nx, int ny, int nz,
                                        double dx, double dy, double dz,
                                        double qom, double u0, double v0, double w0,
                                        double uth0, double vth0, double wth0, int npc,
                                        double* d_harvest_array, double* d_theta_array) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCells = nx * ny * nz;

    if (threadId >= totalCells) return;

    // Convert linear index to 3D coordinates
    int i = threadId / (ny * nz);
    int j = (threadId / nz) % ny;
    int k = threadId % nz;

    int cellIndex = threadId;

    // Process particles in each cell
    for (int ii = 0; ii < npcx; ++ii) {
        for (int jj = 0; jj < npcy; ++jj) {
            for (int kk = 0; kk < npcz; ++kk) {
                int counter = cellIndex * npcx * npcy * npcz
                           + ii * npcy * npcz + jj * npcz + kk;

                // Index for random arrays
                int random_index = counter * 2;

                // Position
                d_rx[counter] = (ii + 0.5) * (dx / npcx) + d_nodeX[i];
                d_ry[counter] = (jj + 0.5) * (dy / npcy) + d_nodeY[j];
                d_rz[counter] = (kk + 0.5) * (dz / npcz) + d_nodeZ[k];

                // Charge
                d_q[counter] = (qom / fabs(qom)) * 
                               (fabs(d_rhos[cellIndex] / npc) * (dx * dy * dz));

                // Velocity vx, vy
                double harvest = d_harvest_array[random_index];
                double prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
                double theta = d_theta_array[random_index];
                d_vx[counter] = u0 + uth0 * prob * cos(theta);
                d_vy[counter] = v0 + vth0 * prob * sin(theta);

                // Velocity vz
                harvest = d_harvest_array[random_index + 1];
                prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
                theta = d_theta_array[random_index + 1];
                d_vz[counter] = w0 + wth0 * prob * cos(theta);

                // ID
                d_ID[counter] = counter;
            }
        }
    }
}

void initRandomNumbersStandard(double* harvest_array, double* theta_array, int is, int npc[], 
                      int npcx[], int npcy[], int npcz[], int nx, int ny, int nz){
  int random_index = 0;
  std::cout<<random_index;

  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)     
      for (int k = 0; k < nz; k++)
        for (int ii = 0; ii < npcx[is]; ii++)
          for (int jj = 0; jj < npcy[is]; jj++)
            for (int kk = 0; kk < npcz[is]; kk++){
              harvest_array[random_index] = rand() / (double)RAND_MAX;
              theta_array[random_index] = 2.0 * M_PI * (rand() / (double)RAND_MAX);
              harvest_array[random_index + 1] = rand() / (double)RAND_MAX;
              theta_array[random_index + 1] = 2.0 * M_PI * (rand() / (double)RAND_MAX);
              random_index += 2;
            }       
}

__global__ void initPartTwostreamsKernel(double* rx, double* ry, double* rz, 
                                        double* q, double* vx, double* vy, double* vz, int* ID,
                                        double* nodeX, double* nodeY, double* nodeZ, double* rhos,
                                        int npcx, int npcy, int npcz,
                                        int nx, int ny, int nz,
                                        double dx, double dy, double dz,
                                        double qom, double u0, double v0, double w0,
                                        double uth0, double vth0, double wth0, int npc,
                                        double* harvest_array, double* theta_array)
{
    // Calcolo dell'indice lineare del thread
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCells = nx * ny * nz; // Numero totale di celle nella griglia 3D

    if (threadId >= totalCells) return; // Evita che i thread oltre il numero di celle facciano qualcosa

    // Converte l'indice lineare del thread nella posizione 3D della cella
    int i = threadId / (ny * nz);           // Coordinata X
    int j = (threadId / nz) % ny;           // Coordinata Y
    int k = threadId % nz;                  // Coordinata Z

    int cellIndex = threadId; // L'indice della cella è uguale al threadId

    double sign = -1.0;

    // Particelle in ciascuna cella
    for (int ii = 0; ii < npcx; ++ii) {
        for (int jj = 0; jj < npcy; ++jj) {
            for (int kk = 0; kk < npcz; ++kk) {
                int counter = cellIndex * npcx * npcy * npcz
                            + ii * npcy * npcz + jj * npcz + kk; // Linearizza indice particelle

                // Calcolo dell'indice per gli array di harvest e theta
                int random_index = counter * 2;

                // Posizione
                rx[counter] = (ii + 0.5) * (dx / npcx) + nodeX[i];
                ry[counter] = (jj + 0.5) * (dy / npcy) + nodeY[j];
                rz[counter] = (kk + 0.5) * (dz / npcz) + nodeZ[k];

                // Carica
                q[counter] = (qom / fabs(qom)) *
                             (fabs(rhos[cellIndex] / npc) * (dx * dy * dz));

                // Velocità vx
                double harvest = harvest_array[random_index];
                double prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
                double theta = theta_array[random_index];
                vx[counter] = u0 * sign + uth0 * prob * cos(theta);

                // Velocità vy
                vy[counter] = v0 + vth0 * prob * sin(theta);

                // Velocità vz
                harvest = harvest_array[random_index + 1];
                prob = sqrt(-2.0 * log(1.0 - 0.999999 * harvest));
                theta = theta_array[random_index + 1];
                vz[counter] = w0 + wth0 * prob * cos(theta);

                // ID
                ID[counter] = counter;

                sign *= -1.0; // Alterna il segno per vx
            }
        }
    }
}

#endif
