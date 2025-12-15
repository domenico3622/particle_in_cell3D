#include <fftw3.h>
#include <math.h>
#include <string.h>
#include "init.hpp"

// FUNZIONI BASE NON TOCCATE ------------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------------------------------------

// FUNZIONI MODIFICATE PER CUDA ---------------------------------------------------------------------------------------

void poisson(double* phi, double* rho, double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int blockX, int blockY) {
    int Nzh = Nz / 2 + 1;
    size_t realSize = Nx * Ny * Nz * sizeof(double);
    size_t complexSize = Nx * Ny * Nzh * sizeof(cufftDoubleComplex);

    double* in;
    cufftDoubleComplex* out;

    // Allocate unified or device memory for in and out
    cudaMalloc(&in, realSize);
    cudaMalloc(&out, complexSize);

    // Copy rho to in
    {
        int size = Nx * Ny * Nz;
        int blockSize = blockX * blockY;
        int gridSize = (size + blockSize - 1) / blockSize;
        copyRhoKernel<<<gridSize, blockSize>>>(in, rho, size);
        cudaDeviceSynchronize();
    }

    // Create forward plan (D2Z)
    cufftHandle forwardPlan;
    cufftPlan3d(&forwardPlan, Nx, Ny, Nz, CUFFT_D2Z);
    cufftExecD2Z(forwardPlan, (cufftDoubleReal*)in, (cufftDoubleComplex*)out);
    cudaDeviceSynchronize();

    // Solve Poisson in Fourier space
    {
        int totalThreadsSolve = Nx * Ny * Nzh;
        int blockSize = blockX * blockY;
        int gridSizeSolve = (totalThreadsSolve + blockSize - 1) / blockSize;
        solvePoissonFourierKernel<<<gridSizeSolve, blockSize>>>(out, Lx, Ly, Lz, Nx, Ny, Nz);
        cudaDeviceSynchronize();
    }

    // Inverse FFT (Z2D)
    cufftHandle inversePlan;
    cufftPlan3d(&inversePlan, Nx, Ny, Nz, CUFFT_Z2D);
    cufftExecZ2D(inversePlan, (cufftDoubleComplex*)out, (cufftDoubleReal*)in);
    cudaDeviceSynchronize();

    // Normalize the solution
    {
        int size = Nx * Ny * Nz;
        int blockSize = blockX * blockY;
        int gridSize = (size + blockSize - 1) / blockSize;
        normalizeKernel<<<gridSize, blockSize>>>(in, Nx, Ny, Nz);
        cudaDeviceSynchronize();
    }

    // Copy in back to phi
    cudaMemcpy(phi, in, realSize, cudaMemcpyDeviceToDevice);

    // Clean up
    cufftDestroy(forwardPlan);
    cufftDestroy(inversePlan);
    cudaFree(in);
    cudaFree(out);
}

void gradient(double *gradX, double *gradY, double *gradZ, double *phi, 
              double Lx, double Ly, double Lz, int Nx, int Ny, int Nz, int sign, int blockX, int blockY) {
    int Nzh = Nz / 2 + 1; // Half-complex size in z

    // Allocate memory for cuFFT
    double *d_phi;
    cufftDoubleComplex *d_out, *d_outX, *d_outY, *d_outZ;
    cudaMalloc(&d_phi, Nx * Ny * Nz * sizeof(double));                       // Real input
    cudaMalloc(&d_out, Nx * Ny * Nzh * sizeof(cufftDoubleComplex));         // Complex Fourier space
    cudaMalloc(&d_outX, Nx * Ny * Nzh * sizeof(cufftDoubleComplex));        // Complex gradient X
    cudaMalloc(&d_outY, Nx * Ny * Nzh * sizeof(cufftDoubleComplex));        // Complex gradient Y
    cudaMalloc(&d_outZ, Nx * Ny * Nzh * sizeof(cufftDoubleComplex));        // Complex gradient Z

    // Copy real input (phi) to device
    cudaMemcpy(d_phi, phi, Nx * Ny * Nz * sizeof(double), cudaMemcpyHostToDevice);

    // Forward FFT: phi -> d_out
    cufftHandle forwardPlan;
    cufftPlan3d(&forwardPlan, Nx, Ny, Nz, CUFFT_D2Z); // Real to complex
    cufftExecD2Z(forwardPlan, d_phi, d_out);
    cudaDeviceSynchronize();

    // Solve Gradients in Fourier Space
    int totalThreads = Nx * Ny * Nzh;
    int blockSize = blockX * blockY;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    solveGradientFourierKernel<<<gridSize, blockSize>>>(d_out, d_outX, d_outY, d_outZ, 
                                                        Lx, Ly, Lz, Nx, Ny, Nz, sign);
    cudaDeviceSynchronize();

    // Inverse FFTs: d_outX, d_outY, d_outZ -> gradX, gradY, gradZ
    cufftHandle inversePlan;
    cufftPlan3d(&inversePlan, Nx, Ny, Nz, CUFFT_Z2D); // Complex to real
    cufftExecZ2D(inversePlan, d_outX, d_phi); // Reuse d_phi to store real gradX
    cudaMemcpy(gradX, d_phi, Nx * Ny * Nz * sizeof(double), cudaMemcpyDeviceToHost);

    cufftExecZ2D(inversePlan, d_outY, d_phi); // Reuse d_phi to store real gradY
    cudaMemcpy(gradY, d_phi, Nx * Ny * Nz * sizeof(double), cudaMemcpyDeviceToHost);

    cufftExecZ2D(inversePlan, d_outZ, d_phi); // Reuse d_phi to store real gradZ
    cudaMemcpy(gradZ, d_phi, Nx * Ny * Nz * sizeof(double), cudaMemcpyDeviceToHost);

    // Normalize Gradients
    int gridSizeNorm = (Nx * Ny * Nz + blockSize - 1) / blockSize;
    normalizeKernel<<<gridSizeNorm, blockSize>>>(gradX, Nx, Ny, Nz);
    normalizeKernel<<<gridSizeNorm, blockSize>>>(gradY, Nx, Ny, Nz);
    normalizeKernel<<<gridSizeNorm, blockSize>>>(gradZ, Nx, Ny, Nz);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_phi);
    cudaFree(d_out);
    cudaFree(d_outX);
    cudaFree(d_outY);
    cudaFree(d_outZ);
    cufftDestroy(forwardPlan);
    cufftDestroy(inversePlan);
}
