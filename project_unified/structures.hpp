#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include<iostream>
using std::cout;

#include <new>
using std::nothrow;

using namespace std;

/* strutture dati usate:
- simu_grid: rappresenta la griglia di simulazione (una griglia 3D).
    - nodeX, nodeY, nodeZ: array che rappresentano le posizioni dei nodi della griglia lungo gli assi x, y e z.
    - minX, maxX, minY, maxY, minZ, maxZ: limiti fisici della griglia lungo ciascun asse.
- simu_fields: contiene i valori del campo elettromagnetico sulla griglia.
    - Contiene array per il potenziale elettrico (phi), la densità di carica totale (rho_tot), i campi elettrici (Exn, Eyn, Ezn) e i campi magnetici (Bxn, Byn, Bzn).
    - Campi magnetici sono definiti sia sui nodi della griglia che nelle celle.
- simu_particles: contiene i dati relativi alle particelle nella simulazione (posizione, velocità, carica, ecc.).
    - ogni particella è rappresentata da: (e sono suddivise in particelle positive e negative)
        - posizione (rx, ry, rz)
        - pelocità (vx, vy, vz)
        - carica (q)
        - ID univoco (ID)
*/

/* valori importanti e il loro significato:
- nx, ny, nz: 
    - numero di celle lungo gli assi x, y e z. Il numero di nodi sarà nx+1, ny+1, nz+1.
- nCells, nNodes:
    - nCells: numero totale di celle nella griglia (nx * ny * nz).
    - nNodes: numero totale di nodi nella griglia ((nx+1) * (ny+1) * (nz+1)).
- ns:
    - Numero di specie di particelle (es. positivo e negativo).
- np:
    - Array che contiene il numero di particelle per specie.
*/

/*! \brief Contains the simulation grid with additional informations
*
* nodeX, nodeY e nodeZ sono array di posizioni dei nodi nella griglia. Ricordati che
* ogni macroparticella ha dei nodi (in questo caso 3D credo ne avremo 8 (?)).
* Esempio: se abbiamo che nx=3 (abbiamo tre celle lungo x), allora avremo
* nx+1=4 nodi, visto che ogni macroparticella dovrà avere 2 nodi lungo x.
* Visivamente sarebbe: |----|----|----| (appunto quattro nodi)
*                      0    1    2    3
* Ora, nodeX potrebbe essere qualcosa del genere: {0.0, 0.25, 0.5, 0.75}.
* Lo stesso discorso vale per le altre dimensioni.
*
* min e max rappresentano i limiti fisici del dominio della griglia lungo
* ciascun asse. Ad esempio:
* - minX: posizione minima lungo l'asse x (es. 0.0).
* - maxX: posizione massima lungo l'asse x (es. 1.0).
* Gli array nodeX, nodeY, nodeZ sono popolati tra questi limiti, per esempio
* dividendo lo spazio equamente.
* 
*/
struct simu_grid
{
  double* nodeX;            /*!< [nx+1]; Linearized x position of grid node. */
  double* nodeY;            /*!< [ny+1]; Linearized y position of grid node. */
  double* nodeZ;            /*!< [nz+1]; Linearized z position of grid node. */
  double minX; double maxX; /*!< min and MAX positions for particles along x. */
  double minY; double maxY; /*!< min and MAX positions for particles along y. */
  double minZ; double maxZ; /*!< min and MAX positions for particles along z. */
};

/*! \brief Contains all field values used in the solver
 */

/*
* 
* qui definiamo il campo elettromagnetico. Nota che nei commenti c'è una svista, perché per
* i campi phi, rho_tot, Exn, Eyn e Ezn dovresti allocare spazio pari al numero totale di nodi,
* quindi (nx+1)*(ny+1)*(nz+1).
* 
* L'unica cosa da capire è che le specie rappresentano che tipo di particelle possiamo trovare nel
* sistema: se vedi config.cfg troverai che abbiamo due specie, praticamente particelle cariche
* positivamente e negativamente.
*/
struct simu_fields
{
  double* phi;       /*!<    [nx*ny*nz]; Linearized electric potential on nodes. */
  double* rho_tot;   /*!<    [nx*ny*nz]; Linearized charge density on nodes. */

  double* Exn;       /*!<    [nx*ny*nz]; Linearized electric field on nodes along x. */
  double* Eyn;       /*!<    [nx*ny*nz]; Linearized electric field on nodes along y. */
  double* Ezn;       /*!<    [nx*ny*nz]; Linearized electric field on nodes along z. */
                         
  double* Bxn;       /*!<    [nx*ny*nz]; Linearized magnetic field on nodes along x. */
  double* Byn;       /*!<    [nx*ny*nz]; Linearized magnetic field on nodes along y. */
  double* Bzn;       /*!<    [nx*ny*nz]; Linearized magnetic field on nodes along z. */
                         
  double* Bxc;       /*!<    [nx*ny*nz]; Linearized magnetic field on cells along x. */
  double* Byc;       /*!<    [nx*ny*nz]; Linearized magnetic field on cells along y. */
  double* Bzc;       /*!<    [nx*ny*nz]; Linearized magnetic field on cells along z. */
                         
  double* rhos;      /*!< [ns*nx*ny*nz]; Linearized charge density per species. */
                          
  double* Jxs;       /*!< [ns*nx*ny*nz]; Linearized current per species along x. */
  double* Jys;       /*!< [ns*nx*ny*nz]; Linearized current per species along y. */
  double* Jzs;       /*!< [ns*nx*ny*nz]; Linearized current per species along z. */
};


/*! \brief Contains all particles.
 *
 * This is (nominally) by far the largest data structure, containing active as well as
 * free particles, with the least amount of data for each particle: position,
 * velocity, and species, and whether this particle is free or not, i.e. active in the
 * domain or available for representing a new particle to be injected in the domain.
 *
 * simu_particles è una struct che contiene posizione, velocità e altri parametri per ogni singola
 * particella nel sistema (non per ogni macroparticella, ma per ogni microparticella). infatti np
 * esprime proprio il numero di microparticelle all'interno del sistema.
 * 
 * nota che nel main avremo un array di puntatori a simu_particles, cioé avremo un array di 2 elementi
 * (le specie sono 2).
 */
struct simu_particles
{
  double* rx;  /*!< [np]; x position. */
  double* ry;  /*!< [np]; y position. */
  double* rz;  /*!< [np]; z position. */

  double* vx;  /*!< [np]; Speed along x. */
  double* vy;  /*!< [np]; Speed along y. */
  double* vz;  /*!< [np]; Speed along z. */

  double* q;   /*!< [np]; Charge. */

  int*  ID;   /*!< [np]; Particle ID. */
};

/* funzioni di allocazione delle struct, niente di nuovo. */
void allocateGrid(simu_grid* * p_grid, int nx, int ny, int nz)
{
  if ((*p_grid = new (nothrow) simu_grid) == nullptr) {cout << "ERROR: Memory allocation failed for grid\n"; exit(EXIT_FAILURE);}
  if (( (*p_grid)->nodeX = new (nothrow) double[nx + 1]) == nullptr) {cout << "ERROR: Memory allocation failed for grid.nodeX\n"; exit(EXIT_FAILURE);}
  if (( (*p_grid)->nodeY = new (nothrow) double[ny + 1]) == nullptr) {cout << "ERROR: Memory allocation failed for grid.nodeY\n"; exit(EXIT_FAILURE);}
  if (( (*p_grid)->nodeZ = new (nothrow) double[nz + 1]) == nullptr) {cout << "ERROR: Memory allocation failed for grid.nodeZ\n"; exit(EXIT_FAILURE);}  
}

void allocateFields(simu_fields* * p_fields, int ns, int nCells, int nNodes)
{
    if ((*p_fields = new (nothrow) simu_fields) == nullptr) {cout << "ERROR: Memory allocation failed for fields\n"; exit(EXIT_FAILURE);}
    if (( (*p_fields)->phi     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.phi\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->rho_tot = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.rho_tot\n"; exit(EXIT_FAILURE);}
    if (( (*p_fields)->Exn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Exn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Eyn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Eyn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Ezn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Ezn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bxn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bxn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Byn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Byn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bzn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bzn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bxc     = new (nothrow) double[nCells]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bxc\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Byc     = new (nothrow) double[nCells]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Byc\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bzc     = new (nothrow) double[nCells]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bzc\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->rhos    = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.rhos\n";    exit(EXIT_FAILURE);}
    if (( (*p_fields)->Jxs     = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.Jxs\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Jys     = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.Jys\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Jzs     = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.Jzs\n";     exit(EXIT_FAILURE);}
}

void allocateParticles(simu_particles** * p_part, int ns, int np[])
{
  if ((*p_part = new (nothrow) simu_particles*[ns]) == nullptr) {cout << "ERROR: Memory allocation failed for part\n"; exit(EXIT_FAILURE);}
  for (int is = 0; is < ns; is++) {
      if (( (*p_part)[is] = new (nothrow) simu_particles) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "]\n"; exit(EXIT_FAILURE);}

      if (( (*p_part)[is]->rx = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].rx\n"; exit(EXIT_FAILURE);}
      if (( (*p_part)[is]->ry = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].ry\n"; exit(EXIT_FAILURE);}
      if (( (*p_part)[is]->rz = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].rz\n"; exit(EXIT_FAILURE);}
      if (( (*p_part)[is]->vx = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].vx\n"; exit(EXIT_FAILURE);}
      if (( (*p_part)[is]->vy = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].vy\n"; exit(EXIT_FAILURE);}
      if (( (*p_part)[is]->vz = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].vz\n"; exit(EXIT_FAILURE);}
      if (( (*p_part)[is]->q  = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].q\n" ; exit(EXIT_FAILURE);}
      if (( (*p_part)[is]->ID = new (nothrow)    int[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].ID\n"; exit(EXIT_FAILURE);}
  }
}

void freeGrid(simu_grid* * p_grid)
{
  delete[] (*p_grid)->nodeX; (*p_grid)->nodeX = nullptr;
  delete[] (*p_grid)->nodeY; (*p_grid)->nodeY = nullptr;
  delete[] (*p_grid)->nodeZ; (*p_grid)->nodeZ = nullptr;
  delete *p_grid;
}

void freeFields(simu_fields* * p_fields)
{
  delete[] (*p_fields)->phi    ; (*p_fields)->phi     = nullptr;
  delete[] (*p_fields)->rho_tot; (*p_fields)->rho_tot = nullptr;
  delete[] (*p_fields)->Exn    ; (*p_fields)->Exn     = nullptr;
  delete[] (*p_fields)->Eyn    ; (*p_fields)->Eyn     = nullptr;
  delete[] (*p_fields)->Ezn    ; (*p_fields)->Ezn     = nullptr;
  delete[] (*p_fields)->Bxn    ; (*p_fields)->Bxn     = nullptr;
  delete[] (*p_fields)->Byn    ; (*p_fields)->Byn     = nullptr;
  delete[] (*p_fields)->Bzn    ; (*p_fields)->Bzn     = nullptr;
  delete[] (*p_fields)->Bxc    ; (*p_fields)->Bxc     = nullptr;
  delete[] (*p_fields)->Byc    ; (*p_fields)->Byc     = nullptr;
  delete[] (*p_fields)->Bzc    ; (*p_fields)->Bzc     = nullptr;
  delete[] (*p_fields)->rhos   ; (*p_fields)->rhos    = nullptr;
  delete[] (*p_fields)->Jxs    ; (*p_fields)->Jxs     = nullptr;
  delete[] (*p_fields)->Jys    ; (*p_fields)->Jys     = nullptr;
  delete[] (*p_fields)->Jzs    ; (*p_fields)->Jzs     = nullptr;
  delete *p_fields;
}

void freeParticles(simu_particles** * p_part, int ns)
{
  for (int is = 0; is < ns; is++) {
      delete[] (*p_part)[is]->rx; (*p_part)[is]->rx = nullptr;
      delete[] (*p_part)[is]->ry; (*p_part)[is]->ry = nullptr;
      delete[] (*p_part)[is]->rz; (*p_part)[is]->rz = nullptr;
      delete[] (*p_part)[is]->vx; (*p_part)[is]->vx = nullptr;
      delete[] (*p_part)[is]->vy; (*p_part)[is]->vy = nullptr;
      delete[] (*p_part)[is]->vz; (*p_part)[is]->vz = nullptr;
      delete[] (*p_part)[is]->q ; (*p_part)[is]->q  = nullptr;
      delete[] (*p_part)[is]->ID; (*p_part)[is]->ID = nullptr;
      delete[] (*p_part)[is];
  }
  delete[] *p_part; *p_part = nullptr;
}
// ------------------------------------------------------------------------------------------------------------

// Unified Memory Allocations
void allocateGrid_gpu(simu_grid** p_grid, int nx, int ny, int nz)
{
    cudaMallocManaged(p_grid, sizeof(simu_grid));
    cudaMallocManaged(&(*p_grid)->nodeX, (nx + 1) * sizeof(double));
    cudaMallocManaged(&(*p_grid)->nodeY, (ny + 1) * sizeof(double));
    cudaMallocManaged(&(*p_grid)->nodeZ, (nz + 1) * sizeof(double));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMallocManaged failed for grid: " 
                << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

void allocateFields_gpu(simu_fields** p_fields, int ns, int nCells, int nNodes)
{
  cudaMallocManaged(p_fields, sizeof(simu_fields));
  cudaMallocManaged(&(*p_fields)->phi, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->rho_tot, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Exn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Eyn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Ezn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Bxn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Byn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Bzn, nNodes * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Bxc, nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Byc, nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Bzc, nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->rhos, ns * nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Jxs, ns * nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Jys, ns * nCells * sizeof(double));
  cudaMallocManaged(&(*p_fields)->Jzs, ns * nCells * sizeof(double));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "ERROR: cudaMallocManaged failed for fields: " 
              << cudaGetErrorString(err) << "\n";
    exit(EXIT_FAILURE);
  }
}

void allocateParticles_gpu(simu_particles*** p_part, int ns, int np[])
{
  cudaMallocManaged(p_part, ns * sizeof(simu_particles*));
  for (int is = 0; is < ns; ++is) {
    cudaMallocManaged(&(*p_part)[is], sizeof(simu_particles));
    cudaMallocManaged(&(*p_part)[is]->rx, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->ry, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->rz, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->vx, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->vy, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->vz, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->q, np[is] * sizeof(double));
    cudaMallocManaged(&(*p_part)[is]->ID, np[is] * sizeof(int));
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "ERROR: cudaMallocManaged failed for parts: " 
              << cudaGetErrorString(err) << "\n";
    exit(EXIT_FAILURE);
  }
}
//------------------------------------------------------------------------------------------------------------
// Standard Memory Allocations with cudaMalloc and cudaMemcpy
void allocateGridStandard(simu_grid* * p_grid, double** d_nodeX, double** d_nodeY, double** d_nodeZ, int nx, int ny, int nz)
{
  if ((*p_grid = new (nothrow) simu_grid) == nullptr) {cout << "ERROR: Memory allocation failed for grid\n"; exit(EXIT_FAILURE);}
  if (( (*p_grid)->nodeX = new (nothrow) double[nx + 1]) == nullptr) {cout << "ERROR: Memory allocation failed for grid.nodeX\n"; exit(EXIT_FAILURE);}
  if (( (*p_grid)->nodeY = new (nothrow) double[ny + 1]) == nullptr) {cout << "ERROR: Memory allocation failed for grid.nodeY\n"; exit(EXIT_FAILURE);}
  if (( (*p_grid)->nodeZ = new (nothrow) double[nz + 1]) == nullptr) {cout << "ERROR: Memory allocation failed for grid.nodeZ\n"; exit(EXIT_FAILURE);}  
  if (cudaMalloc(d_nodeX, (nx + 1) * sizeof(double)) != cudaSuccess) {
        std::cerr << "ERROR: Memory allocation failed for grid.nodeX\n";
        exit(EXIT_FAILURE);
    }
  if (cudaMalloc(d_nodeY, (ny + 1) * sizeof(double)) != cudaSuccess) {
      cout << "ERROR: CUDA memory allocation failed for d_nodeY\n";
      exit(EXIT_FAILURE);
  }
  if (cudaMalloc(d_nodeZ, (nz + 1) * sizeof(double)) != cudaSuccess) {
      cout << "ERROR: CUDA memory allocation failed for d_nodeZ\n";
      exit(EXIT_FAILURE);
  }
}
void allocateFieldsStandard(simu_fields** p_fields, double** d_phi, double** d_rho_tot, double** d_Exn, double** d_Eyn, double** d_Ezn, 
                    double** d_Bxn, double** d_Byn, double** d_Bzn, double** d_Bxc, double** d_Byc, double** d_Bzc, 
                    double** d_rhos, double** d_Jxs, double** d_Jys, double** d_Jzs, int ns, int nCells, int nNodes){
    if ((*p_fields = new (nothrow) simu_fields) == nullptr) {cout << "ERROR: Memory allocation failed for fields\n"; exit(EXIT_FAILURE);}
    if (( (*p_fields)->phi     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.phi\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->rho_tot = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.rho_tot\n"; exit(EXIT_FAILURE);}
    if (( (*p_fields)->Exn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Exn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Eyn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Eyn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Ezn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Ezn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bxn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bxn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Byn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Byn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bzn     = new (nothrow) double[nNodes]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bzn\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bxc     = new (nothrow) double[nCells]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bxc\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Byc     = new (nothrow) double[nCells]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Byc\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Bzc     = new (nothrow) double[nCells]) == nullptr)    {cout << "ERROR: Memory allocation failed for fields.Bzc\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->rhos    = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.rhos\n";    exit(EXIT_FAILURE);}
    if (( (*p_fields)->Jxs     = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.Jxs\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Jys     = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.Jys\n";     exit(EXIT_FAILURE);}
    if (( (*p_fields)->Jzs     = new (nothrow) double[ns*nCells]) == nullptr) {cout << "ERROR: Memory allocation failed for fields.Jzs\n";     exit(EXIT_FAILURE);}
   // Allocazione memoria sul device (GPU) per gli array corrispondenti
    cudaMalloc(d_phi,     nNodes * sizeof(double));
    cudaMalloc(d_rho_tot, nNodes * sizeof(double));
    cudaMalloc(d_Exn,     nNodes * sizeof(double));
    cudaMalloc(d_Eyn,     nNodes * sizeof(double));
    cudaMalloc(d_Ezn,     nNodes * sizeof(double));
    cudaMalloc(d_Bxn,     nNodes * sizeof(double));
    cudaMalloc(d_Byn,     nNodes * sizeof(double));
    cudaMalloc(d_Bzn,     nNodes * sizeof(double));
    cudaMalloc(d_Bxc,     nCells * sizeof(double));
    cudaMalloc(d_Byc,     nCells * sizeof(double));
    cudaMalloc(d_Bzc,     nCells * sizeof(double));
    cudaMalloc(d_rhos,    ns * nCells * sizeof(double));
    cudaMalloc(d_Jxs,     ns * nCells * sizeof(double));
    cudaMalloc(d_Jys,     ns * nCells * sizeof(double));
    cudaMalloc(d_Jzs,     ns * nCells * sizeof(double));
    // Verifica che le allocazioni sulla device siano avvenute correttamente
    if (cudaGetLastError() != cudaSuccess) {
        cout << "ERROR: CUDA memory allocation failed for one or more fields\n";
        exit(EXIT_FAILURE);
    }
}
void allocateParticlesStandard(simu_particles** * p_part, double** d_rx, double** d_ry, double** d_rz,
                       double** d_vx, double** d_vy, double** d_vz, double** d_q, int** d_ID,
                       int ns, int np[])
    {
    if ((*p_part = new (nothrow) simu_particles*[ns]) == nullptr) {cout << "ERROR: Memory allocation failed for part\n"; exit(EXIT_FAILURE);}
    int totalParticles = 0;
    for (int is = 0; is < ns; is++) {
        totalParticles += np[is];
        if (( (*p_part)[is] = new (nothrow) simu_particles) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "]\n"; exit(EXIT_FAILURE);} 
        if (( (*p_part)[is]->rx = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].rx\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->ry = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].ry\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->rz = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].rz\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->vx = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].vx\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->vy = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].vy\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->vz = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].vz\n"; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->q  = new (nothrow) double[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].q\n" ; exit(EXIT_FAILURE);}
        if (( (*p_part)[is]->ID = new (nothrow)    int[np[is]]) == nullptr) {cout << "ERROR: Memory allocation failed for part[" << is << "].ID\n"; exit(EXIT_FAILURE);}
    }
    cudaMalloc(d_rx, totalParticles * sizeof(double));
    cudaMalloc(d_ry, totalParticles * sizeof(double));
    cudaMalloc(d_rz, totalParticles * sizeof(double));
    cudaMalloc(d_vx, totalParticles * sizeof(double));
    cudaMalloc(d_vy, totalParticles * sizeof(double));
    cudaMalloc(d_vz, totalParticles * sizeof(double));
    cudaMalloc(d_q,  totalParticles * sizeof(double));
    cudaMalloc(d_ID, totalParticles * sizeof(int));
    if (cudaGetLastError() != cudaSuccess) {
        cout << "ERROR: CUDA memory allocation failed for one or more particle arrays\n";
        exit(EXIT_FAILURE);
    }
}

void allocateRandomArraysStandard(double**& harvest_array, double**& theta_array, 
                          double*& d_harvest_array, double*& d_theta_array, 
                          int ns, int nx, int ny, int nz, 
                          int npcx[], int npcy[], int npcz[]) {

    // Allocazione memoria sulla host (CPU) per i puntatori agli array
    harvest_array = new (nothrow) double*[ns];
    theta_array = new (nothrow) double*[ns];
    
    if (!harvest_array || !theta_array) {
        cout << "ERROR: Memory allocation failed for harvest_array or theta_array\n";
        exit(EXIT_FAILURE);
    }

    int size=0;
    // Allocazione memoria sulla host (CPU) per ogni specie
    for (int is = 0; is < ns; ++is) {
        size_t totalSize = 2 * npcx[is] * npcy[is] * npcz[is] * nx * ny * nz;
        size+=totalSize;
        // Allocazione memoria per harvest_array[is] e theta_array[is] sulla memoria host
        harvest_array[is] = new (nothrow) double[totalSize];
        theta_array[is] = new (nothrow) double[totalSize];

        if (!harvest_array[is] || !theta_array[is]) {
            cout << "ERROR: Memory allocation failed for harvest_array[" << is << "] or theta_array[" << is << "]\n";
            exit(EXIT_FAILURE);
        }
    }

    cudaMalloc(&d_harvest_array, size * sizeof(double));
    cudaMalloc(&d_theta_array, size * sizeof(double));
    if (cudaGetLastError() != cudaSuccess) {
        cout << "ERROR: CUDA memory allocation failed for harvest_array or theta_array\n";
        exit(EXIT_FAILURE);
    }
}

//------------------------------------------------------------------------------------------------------------
//Free gpu memory
void freeGrid_gpu(simu_grid** p_grid)
{
  cudaFree((*p_grid)->nodeX);
  cudaFree((*p_grid)->nodeY);
  cudaFree((*p_grid)->nodeZ);
  cudaFree(*p_grid);
}

void freeFields_gpu(simu_fields** p_fields)
{
  cudaFree((*p_fields)->phi);
  cudaFree((*p_fields)->rho_tot);
  cudaFree((*p_fields)->Exn);
  cudaFree((*p_fields)->Eyn);
  cudaFree((*p_fields)->Ezn);
  cudaFree((*p_fields)->Bxn);
  cudaFree((*p_fields)->Byn);
  cudaFree((*p_fields)->Bzn);
  cudaFree((*p_fields)->Bxc);
  cudaFree((*p_fields)->Byc);
  cudaFree((*p_fields)->Bzc);
  cudaFree((*p_fields)->rhos);
  cudaFree((*p_fields)->Jxs);
  cudaFree((*p_fields)->Jys);
  cudaFree((*p_fields)->Jzs);
  cudaFree(*p_fields);
}

void freeParticles_gpu(simu_particles*** p_part, int ns)
{
  for (int is = 0; is < ns; ++is) {
    cudaFree((*p_part)[is]->rx);
    cudaFree((*p_part)[is]->ry);
    cudaFree((*p_part)[is]->rz);
    cudaFree((*p_part)[is]->vx);
    cudaFree((*p_part)[is]->vy);
    cudaFree((*p_part)[is]->vz);
    cudaFree((*p_part)[is]->q);
    cudaFree((*p_part)[is]->ID);
    cudaFree((*p_part)[is]);
  }
  cudaFree(*p_part);
}
//-------------------------------------------------------------
//free standard
void freeGridStandard(simu_grid** p_grid, double** d_nodeX, double** d_nodeY, double** d_nodeZ)
{
    delete[] (*p_grid)->nodeX; (*p_grid)->nodeX = nullptr;
    delete[] (*p_grid)->nodeY; (*p_grid)->nodeY = nullptr;
    delete[] (*p_grid)->nodeZ; (*p_grid)->nodeZ = nullptr;
    cudaFree(d_nodeX);
    cudaFree(d_nodeY);
    cudaFree(d_nodeZ);
    delete *p_grid;
}

void freeFieldsStandard(simu_fields** p_fields, double** d_phi, double** d_rho_tot, double** d_Exn, double** d_Eyn, double** d_Ezn, 
                double** d_Bxn, double** d_Byn, double** d_Bzn, double** d_Bxc, double** d_Byc, double** d_Bzc,
                double** d_rhos, double** d_Jxs, double** d_Jys, double** d_Jzs)
{
    // Libera la memoria sulla host (CPU)
    delete[] (*p_fields)->phi; (*p_fields)->phi = nullptr;
    delete[] (*p_fields)->rho_tot; (*p_fields)->rho_tot = nullptr;
    delete[] (*p_fields)->Exn; (*p_fields)->Exn = nullptr;
    delete[] (*p_fields)->Eyn; (*p_fields)->Eyn = nullptr;
    delete[] (*p_fields)->Ezn; (*p_fields)->Ezn = nullptr;
    delete[] (*p_fields)->Bxn; (*p_fields)->Bxn = nullptr;
    delete[] (*p_fields)->Byn; (*p_fields)->Byn = nullptr;
    delete[] (*p_fields)->Bzn; (*p_fields)->Bzn = nullptr;
    delete[] (*p_fields)->Bxc; (*p_fields)->Bxc = nullptr;
    delete[] (*p_fields)->Byc; (*p_fields)->Byc = nullptr;
    delete[] (*p_fields)->Bzc; (*p_fields)->Bzc = nullptr;
    delete[] (*p_fields)->rhos; (*p_fields)->rhos = nullptr;
    delete[] (*p_fields)->Jxs; (*p_fields)->Jxs = nullptr;
    delete[] (*p_fields)->Jys; (*p_fields)->Jys = nullptr;
    delete[] (*p_fields)->Jzs; (*p_fields)->Jzs = nullptr;

    // Libera la memoria sulla device (GPU)
    cudaFree(d_phi); 
    cudaFree(d_rho_tot);
    cudaFree(d_Exn); 
    cudaFree(d_Eyn);
    cudaFree(d_Ezn);
    cudaFree(d_Bxn);
    cudaFree(d_Byn);
    cudaFree(d_Bzn);
    cudaFree(d_Bxc);
    cudaFree(d_Byc);
    cudaFree(d_Bzc);
    cudaFree(d_rhos);
    cudaFree(d_Jxs);
    cudaFree(d_Jys);
    cudaFree(d_Jzs);

    // Libera la struttura sulla host
    delete *p_fields;
}


void freeParticlesStandard(simu_particles** *p_part, int ns, double** d_rx, double** d_ry, double** d_rz, 
                   double** d_vx, double** d_vy, double** d_vz, double** d_q, int** d_ID)
{
    for (int is = 0; is < ns; is++) {
        // Libera la memoria sulla host (CPU)
        delete[] (*p_part)[is]->rx; (*p_part)[is]->rx = nullptr;
        delete[] (*p_part)[is]->ry; (*p_part)[is]->ry = nullptr;
        delete[] (*p_part)[is]->rz; (*p_part)[is]->rz = nullptr;
        delete[] (*p_part)[is]->vx; (*p_part)[is]->vx = nullptr;
        delete[] (*p_part)[is]->vy; (*p_part)[is]->vy = nullptr;
        delete[] (*p_part)[is]->vz; (*p_part)[is]->vz = nullptr;
        delete[] (*p_part)[is]->q;  (*p_part)[is]->q  = nullptr;
        delete[] (*p_part)[is]->ID; (*p_part)[is]->ID = nullptr;
    }

    // Libera la memoria sulla device (GPU)
    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_rz);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_q);
    cudaFree(d_ID);

    // Libera la memoria della struttura sulla host
    delete[] *p_part;
    *p_part = nullptr;
}
#endif