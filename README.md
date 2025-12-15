# CUDA Parallelization of 3D Particle-in-Cell (PIC) Simulation

## Project Overview
This project focuses on the parallelization and optimization of a 3D Particle-in-Cell (PIC) simulation using CUDA. The goal is to accelerate the computationally intensive steps of the simulation—specifically the particle mover and the field solver—by leveraging the massive parallelism of NVIDIA GPUs. 

The project involves two distinct implementations to explore different memory management strategies and optimization techniques:
1.  **Standard Implementation (`project_standard`)**: Uses explicit memory management.
2.  **Unified Implementation (`project_unified`)**: Uses CUDA Unified Memory.

Both implementations aim to solve the Vlasov-Maxwell system for plasma physics simulations, accelerating the **Interpolation (Scatter/Gather)**, **Particle Pusher**, and **Field Solver** phases.

---

## Project Structure

The repository represents a complete CUDA-based 3D PIC code, organized into the following key directories:

### 1. `project_standard` (Explicit Memory)
This version employs a traditional CUDA programming model with **Explicit Memory Management**:
*   **Memory Handling**: Manually allocates memory on both Host (CPU) and Device (GPU) using `malloc`/`cudaMalloc`. Data transfer between Host and Device is explicitly managed via `cudaMemcpy`.
*   **Data Structures**: Adopts a "Structure of Arrays" (SoA) approach where particle properties (e.g., `rx`, `ry`, `vx`, `vy`) are passed as separate arrays to CUDA kernels to ensure coalesced memory access.
*   **Kernels**: Kernels accept individual device pointers (e.g., `double* d_rx`, `double* d_vx`).

### 2. `project_unified` (Unified Memory)
This version leverages **CUDA Unified Memory (Managed Memory)** to simplify memory management and data access:
*   **Memory Handling**: Uses `cudaMallocManaged` to allocate memory that is accessible from both the CPU and GPU. The CUDA driver handles the migration of pages on demand.
*   **Data Structures**: Passes complex structures (e.g., `simu_particles` structs) directly to kernels.
*   **Kernels**: Kernels accept pointers to structures (e.g., `simu_particles** part`), allowing for cleaner code that resembles the serial C++ implementation while running on the GPU.

---

## Technical Implementations & Features

### Core Algorithm: Particle-in-Cell (PIC) Cycle
The simulation follows the standard PIC cycle, fully accelerated on the GPU:
1.  **Interpolation (Particles $\to$ Grid)**: 
    *   Particles deposit charge (`q`) onto the grid nodes (`rho`).
    *   **Implementation**: Utilizes `atomicAdd` to handle race conditions where multiple particles contribute to the same grid cell.
2.  **Maxwell Solver (Field Solve)**:
    *   Solves the Poisson equation and computes gradients ($\phi \to \vec{E}$).
    *   **Implementation**: Accelerated using **cuFFT** (CUDA Fast Fourier Transform library) to solve the Poisson equation in the spectral domain.
3.  **Particle Pusher**:
    *   Updates particle velocities (`vx`, `vy`, `vz`) using the Lorentz force from grid fields.
    *   Refreshes particle positions (`rx`, `ry`, `rz`) and handles periodic boundary conditions.
    *   **Implementation**: Fully parallelized kernels (`updateParticlePositionKernel`, `updateParticleVelocityKernel`) with one thread per particle.

### Optimizations Explored
Several optimization strategies were implemented and tested across both versions:
*   **Atomic Operations**: Essential for the Scatter phase to correctly accumulate charge density from thousands of threads efficiently.
*   **cuFFT Integration**: Replaced the iterative or serial field solvers with highly optimized FFT-based solvers on the GPU.
*   **Privatization (Explored)**: Investigated using thread-local or shared memory buffers to accumulate partial charge densities before writing to global memory, aiming to reduce contention on global atomic operations.
*   **Coarsening (Explored)**: Investigated assigning multiple particles to a single thread to increase instruction-level parallelism and hide memory latency.

---

## How to Build and Run

### Prerequisites
*   **OS**: Linux / Windows (with WSL)
*   **Compiler**: `nvcc` (NVIDIA CUDA Compiler)
*   **Libraries**: `cufft` (NVIDIA CUDA Fast Fourier Transform library)

### Compilation
Each implementation has its own `Makefile`. Navigate to the desired directory and run `make`:

```bash
# For Standard Implementation
cd project_standard
make

# For Unified Implementation
cd project_unified
make
```

### Execution
Run the compiled executable. You may need to provide block size arguments (e.g., `<blockX> <blockY>`).

```bash
# Example
./simplePIC3D_standard 16 16
```

---

## Results and Performance

The parallelization yields significant performance improvements over the serial CPU version, particularly for large numbers of particles and grid cells. 

*   **Speedup**: The GPU acceleration dramatically reduces the time per time-step.
*   **Scalability**: The `atomicAdd` and cuFFT approaches scale well with problem size, though the Scatter phase remains memory-bound due to random access patterns.
*   **Comparison**: The "Standard" version typically offers finer control over data movement, while the "Unified" version offers ease of development.

**For detailed performance metrics, speedup graphs, and a comprehensive analysis of the results (including comparisons between optimization strategies), please refer to the project Report.**

---

## References
*   **PIC.pdf**: Theoretical background on the Particle-in-Cell method used in this project.
*   **Report**: Detailed analysis of implementation choices and performance results.
