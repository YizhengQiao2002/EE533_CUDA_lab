// conv_lib.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned int uint;


#ifndef TILE
#define TILE 16
#endif

__global__ void conv2d_gpu_kernel(
    const unsigned int *d_image,
    const int *d_kernel,
    unsigned int *d_output,
    int M,
    int N,
    int normalize_divisor
) {
    int pad = N / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int out_x = bx * TILE + tx;
    int out_y = by * TILE + ty;

    extern __shared__ int s_mem[]; 
    int S = TILE + 2*pad;

    int base_x = bx * TILE - pad;
    int base_y = by * TILE - pad;

    for (int yy = ty; yy < S; yy += blockDim.y) {
        for (int xx = tx; xx < S; xx += blockDim.x) {
            int gx = base_x + xx;
            int gy = base_y + yy;
            int sval = 0;
            if (gx >= 0 && gx < M && gy >= 0 && gy < M) sval = (int)d_image[gy * M + gx];
            s_mem[yy * S + xx] = sval;
        }
    }
    __syncthreads();

    if (out_x < M && out_y < M) {
        int acc = 0;
        for (int ky = 0; ky < N; ky++) {
            for (int kx = 0; kx < N; kx++) {
                int sx = tx + kx;
                int sy = ty + ky;
                int sim = s_mem[sy * S + sx];
                int kval = d_kernel[ky * N + kx];
                acc += sim * kval;
            }
        }
        if (normalize_divisor > 0) acc /= normalize_divisor;
        if (acc < 0) acc = 0;
        if (acc > 255) acc = 255;
        d_output[out_y * M + out_x] = (unsigned int)acc;
    }
}

extern "C" __declspec(dllexport)
void gpu_convolve(
    unsigned int *h_image, 
    int M,
    int *h_kernel,        
    int N,
    unsigned int *h_output 
) {
    size_t img_bytes = sizeof(uint) * (size_t)M * M;
    size_t ker_bytes = sizeof(int) * (size_t)N * N;
    uint *d_image = nullptr;
    int *d_kernel = nullptr;
    uint *d_output = nullptr;

    cudaMalloc(&d_image, img_bytes);
    cudaMalloc(&d_kernel, ker_bytes);
    cudaMalloc(&d_output, img_bytes);

    // H2D
    cudaMemcpy(d_image, h_image, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, ker_bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid( (M + TILE - 1) / TILE, (M + TILE - 1) / TILE );
    int S = TILE + 2*(N/2);
    size_t shared_bytes = sizeof(int) * S * S;

    conv2d_gpu_kernel<<<grid, block, shared_bytes>>>(d_image, d_kernel, d_output, M, N, (N*N));
    cudaDeviceSynchronize();

    // D2H
    cudaMemcpy(h_output, d_output, img_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
