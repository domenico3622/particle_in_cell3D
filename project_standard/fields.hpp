#include <fftw3.h>
#include <math.h>
#include <string.h>
#include <cufft.h>

// FUNZIONI BASE MODIFICATE PER CUFFT -------------------------------------------------------------------------------------
namespace depth_level_2
{
    void pad(double *in, double *rho, int Nx, int Ny, int Nzh)
    {
        int Nz = 2 * Nzh - 2;
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++)
                for (int k = 0; k < 2 * Nzh; k++)
                    if (k < Nz)
                        in[i * 2 * Nzh * Ny + j * 2 * Nzh + k] = rho[i * Nz * Ny + j * Nz + k];
                    else
                        in[i * 2 * Nzh * Ny + j * 2 * Nzh + k] = 0;
    }

    void unpad(double *phi, double *in, int Nx, int Ny, int Nzh)
    {
        int Nz = 2 * Nzh - 2;
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++)
                for (int k = 0; k < 2 * Nzh; k++)
                    if (k < Nz)
                        phi[i * Nz * Ny + j * Nz + k] = in[i * 2 * Nzh * Ny + j * 2 * Nzh + k];
    }

    void normalize(double *phi, int Nx, int Ny, int Nz)
    {
        for (int i = 0; i < Nx; i++)
            for (int j = 0; j < Ny; j++)
                for (int k = 0; k < Nz; k++)
                    phi[i * Nz * Ny + j * Nz + k] /= Nx * Ny * Nz;
    }

    void check_avg_sigma_tot(double& avg, double& sigma, double* rho_tot, int nx, int ny, int nz)
    {
        avg = 0.0;
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)     
              for (int k = 0; k < nz; k++)
                avg += rho_tot[i*ny*nz + j*nz + k];
        avg /= (nx*ny*nz);

        sigma = 0.0;
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)     
              for (int k = 0; k < nz; k++)
                sigma += pow(rho_tot[i*ny*nz + j*nz + k] - avg, 2);
        sigma = sqrt(sigma/(nx*ny*nz));
    }

    void solvePoissonFourier(fftw_complex *out, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz)
    {
        int Nzh = Nz / 2 + 1;
        int II, JJ;
        double k1, k2, k3;
        for (int i = 0; i < Nx; i++)
        {
            if (2 * i < Nx)
                II = i;
            else
                II = Nx - i;
            k1 = 2 * M_PI * II / Lx;

            for (int j = 0; j < Ny; j++)
            {
                if (2 * j < Ny)
                    JJ = j;
                else
                    JJ = Ny - j;
                k2 = 2 * M_PI * JJ / Ly;

                for (int k = 0; k < Nzh; k++)
                {
                    k3 = 2 * M_PI * k / Lz;
                    double fac = -1.0 * (pow(k1, 2) + pow(k2, 2) + pow(k3, 2));
                    
                    if (fabs(fac) < 1e-14) {
                        out[k + Nzh * (j + Ny * i)][0] = 0.0;
                        out[k + Nzh * (j + Ny * i)][1] = 0.0;
                    }
                    else {
                        out[k + Nzh * (j + Ny * i)][0] /= fac;
                        out[k + Nzh * (j + Ny * i)][1] /= fac;
                    }
                }
            }
        }
    }

    void solveGradientFourier(fftw_complex *out, fftw_complex *outX, fftw_complex *outY, fftw_complex *outZ, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int sign)
    {
        int Nzh = Nz / 2 + 1;
        int II, JJ;
        double k1, k2, k3;
        for (int i = 0; i < Nx; i++)
        {
            if (2 * i < Nx)
                II = i;
            else
                II = -(Nx - i);
            k1 = 2 * M_PI * II / Lx;

            for (int j = 0; j < Ny; j++)
            {
                if (2 * j < Ny)
                    JJ = j;
                else
                    JJ = -(Ny - j);
                k2 = 2 * M_PI * JJ / Ly;

                for (int k = 0; k < Nzh; k++)
                {
                    k3 = 2 * M_PI * k / Lz;
                    double fac = -1.0 * (pow(k1, 2) + pow(k2, 2) + pow(k3, 2));

                    if (fabs(fac) < 1e-14)
                    {
                        outX[k + Nzh * (j + Ny * i)][0] = outY[k + Nzh * (j + Ny * i)][0] = outZ[k + Nzh * (j + Ny * i)][0] = 0.0;
                        outX[k + Nzh * (j + Ny * i)][1] = outY[k + Nzh * (j + Ny * i)][1] = outZ[k + Nzh * (j + Ny * i)][1] = 0.0;
                    }
                    else
                    {                    
                        outX[k + Nzh * (j + Ny * i)][0] = -out[k + Nzh * (j + Ny * i)][1] * k1 * sign;
                        outX[k + Nzh * (j + Ny * i)][1] =  out[k + Nzh * (j + Ny * i)][0] * k1 * sign;

                        outY[k + Nzh * (j + Ny * i)][0] = -out[k + Nzh * (j + Ny * i)][1] * k2* sign;
                        outY[k + Nzh * (j + Ny * i)][1] =  out[k + Nzh * (j + Ny * i)][0] * k2* sign;

                        outZ[k + Nzh * (j + Ny * i)][0] = -out[k + Nzh * (j + Ny * i)][1] * k3* sign;
                        outZ[k + Nzh * (j + Ny * i)][1] =  out[k + Nzh * (j + Ny * i)][0] * k3* sign;
                    }
                }
            }
        }
    }

} // end of depth_level_2 namespace

// KERNELS CUDA ----------------------------------------------------------------------------------------------------------
__global__ void computeRhoTotKernel(double* d_rho_tot, double* d_rhos, int ns, int nx, int ny, int nz) {
    // Calcolo dell'indice globale del thread
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Numero totale di celle nella griglia 3D
    int totalCells = nx * ny * nz;

    // Verifica che il thread sia nel range corretto
    if (threadId >= totalCells) return;

    // Calcolo delle coordinate i, j, k a partire dall'indice lineare
    int i = threadId / (ny * nz);           // Coordinata X
    int j = (threadId / nz) % ny;           // Coordinata Y
    int k = threadId % nz;                  // Coordinata Z

    // Calcola l'indice lineare per rho_tot
    int idx = i * ny * nz + j * nz + k;

    // Somma il contributo di tutte le specie (ns)
    for (int is = 0; is < ns; is++) {
        int idx_rhos = is * nx * ny * nz + idx; // Indice per rhos con specie
        d_rho_tot[idx] += -4.0 * M_PI * d_rhos[idx_rhos];
    }
}

void computeRhoTot(double* rho_tot, double* rhos, int ns, int nx, int ny, int nz)
{
    for (int is = 0; is < ns; is++)
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)     
              for (int k = 0; k < nz; k++)
                rho_tot[i*ny*nz + j*nz + k] += - 4.0 * M_PI * rhos[is*nx*ny*nz + i*ny*nz + j*nz + k];  
}

void poisson(double *phi, double *rho, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz)
{
    /* <<< use cuFFT https://docs.nvidia.com/cuda/cufft/ instead of FFTW3 >>> */

    fftw_complex *mem;
    fftw_complex *out;
    double *in;

    int Nzh = Nz / 2 + 1;

    mem = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nzh);
    memset((unsigned char *)mem, 0, sizeof(fftw_complex) * Nx * Ny * Nzh); // <<< use cudaMemset >>>

    out = mem;
    in = mem[0];

    fftw_plan fwrd = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, in, out, FFTW_MEASURE);
    fftw_plan bwrd = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, out, in, FFTW_MEASURE);

    depth_level_2::pad(in, rho, Nx, Ny, Nzh); // <<< convert to a cuda kernel >>>

    fftw_execute(fwrd);

    depth_level_2::solvePoissonFourier(out, Lx, Ly, Lz, Nx, Ny, Nz); // <<< convert to a cuda kernel >>>

    fftw_execute(bwrd);

    depth_level_2::unpad(phi, in, Nx, Ny, Nzh); // <<< convert to a cuda kernel >>>
    depth_level_2::normalize(phi, Nx, Ny, Nz);  // <<< convert to a cuda kernel >>>
}

// Kernel to copy rho into in (if rho and in are both on GPU)
__global__ void copyRhoKernel(double* in, const double* rho, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        in[tid] = rho[tid];
    }
}

// Kernel to normalize the output array
__global__ void normalizeKernel(double* phi, int Nx, int Ny, int Nz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int size = Nx * Ny * Nz;
    if (tid < size) {
        phi[tid] /= (double)(Nx * Ny * Nz);
    }
}

// Kernel to solve Poisson in Fourier space
__global__ void solvePoissonFourierKernel(cufftDoubleComplex* out, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz) {
    int Nzh = Nz/2 + 1;
    int size = Nx * Ny * Nzh;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    int i = tid / (Ny * Nzh);
    int remainder = tid % (Ny * Nzh);
    int j = remainder / Nzh;
    int k = remainder % Nzh;

    int II = (2*i < Nx) ? i : Nx - i;
    int JJ = (2*j < Ny) ? j : Ny - j;
    double k1 = 2.0 * M_PI * II / Lx;
    double k2 = 2.0 * M_PI * JJ / Ly;
    double k3 = 2.0 * M_PI * k  / Lz;

    double fac = -(k1*k1 + k2*k2 + k3*k3);

    cufftDoubleComplex val = out[tid];
    if (fabs(fac) < 1e-14) {
        out[tid].x = 0.0;
        out[tid].y = 0.0;
    } else {
        out[tid].x = val.x / fac;
        out[tid].y = val.y / fac;
    }
}

__global__ void solveGradientFourierKernel(cufftDoubleComplex *out, cufftDoubleComplex *outX, 
                                           cufftDoubleComplex *outY, cufftDoubleComplex *outZ, 
                                           double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int sign) {
    int Nzh = Nz / 2 + 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Nx * Ny * Nzh) return;

    int i = tid / (Ny * Nzh);
    int remainder = tid % (Ny * Nzh);
    int j = remainder / Nzh;
    int k = remainder % Nzh; //per la coalescenza, k deve essere l'indice più veloce
    //cosi facendo, k è l'indice più veloce, j quello intermedio e i il più lento, questo permette ai thread 
    //di leggere dati contigui in memoria

    int II = (2 * i < Nx) ? i : i - Nx;
    int JJ = (2 * j < Ny) ? j : j - Ny;
    double k1 = 2.0 * M_PI * II / Lx;
    double k2 = 2.0 * M_PI * JJ / Ly;
    double k3 = 2.0 * M_PI * k / Lz;

    cufftDoubleComplex value = out[tid];

    // Gradient computation
    outX[tid].x = -value.y * k1 * sign;
    outX[tid].y =  value.x * k1 * sign;

    outY[tid].x = -value.y * k2 * sign;
    outY[tid].y =  value.x * k2 * sign;

    outZ[tid].x = -value.y * k3 * sign;
    outZ[tid].y =  value.x * k3 * sign;
}

void poissonCUFFT(double* d_phi, double* d_rho, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int blockX, int blockY)
{
    int Nzh = Nz / 2 + 1;
    size_t realSize = Nx * Ny * Nz * sizeof(double);
    size_t complexSize = Nx * Ny * Nzh * sizeof(cufftDoubleComplex);

    // Allocazione della memoria per d_in e d_out sulla GPU
    double* d_in;
    cufftDoubleComplex* d_out;
    cudaMalloc(&d_in, realSize);
    cudaMalloc(&d_out, complexSize);

    // Copia di d_rho in d_in
    cudaMemcpy(d_in, d_rho, realSize, cudaMemcpyDeviceToDevice);

    // Creazione dei piani CUFFT
    cufftHandle forwardPlan;
    cufftPlan3d(&forwardPlan, Nx, Ny, Nz, CUFFT_D2Z);
    cufftExecD2Z(forwardPlan, d_in, d_out);
    cudaDeviceSynchronize();

    // Risoluzione dell'equazione di Poisson nello spazio di Fourier
    int totalThreads = Nx * Ny * Nzh;
    int blockSize = blockX * blockY;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    solvePoissonFourierKernel<<<gridSize, blockSize>>>(d_out, Lx, Ly, Lz, Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // Esecuzione della FFT inversa
    cufftHandle inversePlan;
    cufftPlan3d(&inversePlan, Nx, Ny, Nz, CUFFT_Z2D);
    cufftExecZ2D(inversePlan, d_out, d_in);
    cudaDeviceSynchronize();

    // Normalizzazione del risultato
    int totalCells = Nx * Ny * Nz;
    gridSize = (totalCells + blockSize - 1) / blockSize;
    normalizeKernel<<<gridSize, blockSize>>>(d_in, Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // Copia del risultato in d_phi
    cudaMemcpy(d_phi, d_in, realSize, cudaMemcpyDeviceToDevice);

    // Pulizia delle risorse
    cufftDestroy(forwardPlan);
    cufftDestroy(inversePlan);
    cudaFree(d_in);
    cudaFree(d_out);
}

void gradient(double *gradX, double *gradY, double *gradZ, double *phi, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int sign)
{
    // <<< use cuFFT https://docs.nvidia.com/cuda/cufft/ instead of FFTW3 >>>

    fftw_complex *mem, *memX, *memY, *memZ;
    fftw_complex *out, *outX, *outY, *outZ;
    double *in, *inX, *inY, *inZ;

    int Nzh = Nz / 2 + 1;

    mem  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nzh);
    memX = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nzh);
    memY = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nzh);
    memZ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * Nx * Ny * Nzh);

    memset((unsigned char *)mem,  0, sizeof(fftw_complex) * Nx * Ny * Nzh);
    memset((unsigned char *)memX, 0, sizeof(fftw_complex) * Nx * Ny * Nzh);
    memset((unsigned char *)memY, 0, sizeof(fftw_complex) * Nx * Ny * Nzh);
    memset((unsigned char *)memZ, 0, sizeof(fftw_complex) * Nx * Ny * Nzh);

    out  = mem;
    outX = memX;
    outY = memY;
    outZ = memZ;
    in  = mem[0];
    inX = memX[0];
    inY = memY[0];
    inZ = memZ[0];

    fftw_plan fwrd  = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, in,   out, FFTW_MEASURE);
    fftw_plan bwrdX = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, outX, inX, FFTW_MEASURE);
    fftw_plan bwrdY = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, outY, inY, FFTW_MEASURE);
    fftw_plan bwrdZ = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, outZ, inZ, FFTW_MEASURE);

    depth_level_2::pad(in, phi, Nx, Ny, Nzh); // <<< use previously converted cuda kernel >>>

    fftw_execute(fwrd);

    depth_level_2::solveGradientFourier(out, outX, outY, outZ, Lx, Ly, Lz, Nx, Ny, Nz, sign); // <<< convert to a cuda kernel >>>

    fftw_execute(bwrdX);
    fftw_execute(bwrdY);
    fftw_execute(bwrdZ);

    depth_level_2::unpad(gradX, inX, Nx, Ny, Nzh); // <<< use previously converted cuda kernel >>>
    depth_level_2::unpad(gradY, inY, Nx, Ny, Nzh); // <<< use previously converted cuda kernel >>>
    depth_level_2::unpad(gradZ, inZ, Nx, Ny, Nzh); // <<< use previously converted cuda kernel >>>

    depth_level_2::normalize(gradX, Nx, Ny, Nz);   // <<< use previously converted cuda kernel >>>
    depth_level_2::normalize(gradY, Nx, Ny, Nz);   // <<< use previously converted cuda kernel >>>
    depth_level_2::normalize(gradZ, Nx, Ny, Nz);   // <<< use previously converted cuda kernel >>>
}

void gradientCUFFT(double *d_gradX, double *d_gradY, double *d_gradZ, double *d_phi, 
                   double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int sign, int blockX, int blockY) {
    int Nzh = Nz / 2 + 1;
    size_t realSize = Nx * Ny * Nz * sizeof(double);
    size_t complexSize = Nx * Ny * Nzh * sizeof(cufftDoubleComplex);

    // Allocazione della memoria per gli array complessi sulla GPU
    cufftDoubleComplex *d_out, *d_outX, *d_outY, *d_outZ;
    cudaMalloc(&d_out,  complexSize);
    cudaMalloc(&d_outX, complexSize);
    cudaMalloc(&d_outY, complexSize);
    cudaMalloc(&d_outZ, complexSize);

    // Creazione dei piani CUFFT
    cufftHandle planForward, planInverse;
    cufftPlan3d(&planForward, Nx, Ny, Nz, CUFFT_D2Z);
    cufftPlan3d(&planInverse, Nx, Ny, Nz, CUFFT_Z2D);

    // Esecuzione della FFT forwards
    cufftExecD2Z(planForward, d_phi, d_out);
    cudaDeviceSynchronize();

    // Calcolo del gradiente nello spazio di Fourier
    int totalThreads = Nx * Ny * Nzh;
    int blockSize = blockX * blockY;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    solveGradientFourierKernel<<<gridSize, blockSize>>>(d_out, d_outX, d_outY, d_outZ, Lx, Ly, Lz, Nx, Ny, Nz, sign);
    cudaDeviceSynchronize();

    // Esecuzione delle FFT inverse
    cufftExecZ2D(planInverse, d_outX, d_gradX);
    cufftExecZ2D(planInverse, d_outY, d_gradY);
    cufftExecZ2D(planInverse, d_outZ, d_gradZ);
    cudaDeviceSynchronize();

    // Normalizzazione dei risultati
    int totalCells = Nx * Ny * Nz;
    gridSize = (totalCells + blockSize - 1) / blockSize;
    normalizeKernel<<<gridSize, blockSize>>>(d_gradX, Nx, Ny, Nz);
    normalizeKernel<<<gridSize, blockSize>>>(d_gradY, Nx, Ny, Nz);
    normalizeKernel<<<gridSize, blockSize>>>(d_gradZ, Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // Pulizia delle risorse
    cufftDestroy(planForward);
    cufftDestroy(planInverse);
    cudaFree(d_out);
    cudaFree(d_outX);
    cudaFree(d_outY);
    cudaFree(d_outZ);
}
