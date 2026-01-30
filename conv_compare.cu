// conv_compare.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

typedef unsigned int uint;


void conv2d_cpu_uint(
    const uint *image,    // M x M
    const int *kernel,    // N x N 
    uint *output,         // M x M
    int M,
    int N,
    int normalize_divisor 
) {
    int pad = N / 2;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int acc = 0;
            for (int ki = 0; ki < N; ki++) {
                for (int kj = 0; kj < N; kj++) {
                    int ii = i + ki - pad;
                    int jj = j + kj - pad;
                    if (ii >= 0 && ii < M && jj >= 0 && jj < M) {
                        acc += (int)image[ii * M + jj] * kernel[ki * N + kj];
                    }
                }
            }
            if (normalize_divisor > 0) acc /= normalize_divisor;
            if (acc < 0) acc = 0;
            if (acc > 255) acc = 255;
            output[i * M + j] = (uint)acc;
        }
    }
}

void generate_random_image(uint *img, int M) {
    for (int i = 0; i < M * M; i++) img[i] = rand() % 256;
}

void save_pgm(const char *filename, const uint *img, int M) {
    FILE *f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Unable to open %s for writing\n", filename); return; }
    fprintf(f, "P5\n%d %d\n255\n", M, M);
    unsigned char *buf = (unsigned char*)malloc(M * M);
    for (int i = 0; i < M * M; i++) {
        int v = img[i];
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        buf[i] = (unsigned char)v;
    }
    fwrite(buf, 1, M * M, f);
    free(buf);
    fclose(f);
}


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

    int out_x = bx * TILE + tx; // column (j)
    int out_y = by * TILE + ty; // row (i)

    extern __shared__ int s_mem[]; 
    int S = TILE + 2*pad;
    int s_idx = ty * S + tx; 


    int base_x = bx * TILE - pad; 
    int base_y = by * TILE - pad; 

    for (int yy = ty; yy < S; yy += blockDim.y) {
        for (int xx = tx; xx < S; xx += blockDim.x) {
            int gx = base_x + xx;
            int gy = base_y + yy;
            int sval = 0;
            if (gx >= 0 && gx < M && gy >= 0 && gy < M) {
                sval = (int)d_image[gy * M + gx];
            } else {
                sval = 0;
            }
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

void conv2d_gpu(
    const uint *h_image,
    const int *h_kernel,
    uint *h_output,
    int M,
    int N,
    int normalize_divisor
) {
    size_t img_bytes = sizeof(uint) * (size_t)M * M;
    size_t ker_bytes = sizeof(int) * (size_t)N * N;

    unsigned int *d_image = nullptr;
    int *d_kernel = nullptr;
    unsigned int *d_output = nullptr;

    cudaMalloc(&d_image, img_bytes);
    cudaMalloc(&d_kernel, ker_bytes);
    cudaMalloc(&d_output, img_bytes);

    cudaMemcpy(d_image, h_image, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, ker_bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid( (M + TILE - 1) / TILE, (M + TILE - 1) / TILE );

    int S = TILE + 2*(N/2);
    size_t shared_bytes = sizeof(int) * S * S;

    conv2d_gpu_kernel<<<grid, block, shared_bytes>>>(d_image, d_kernel, d_output, M, N, normalize_divisor);

    cudaMemcpy(h_output, d_output, img_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

void conv2d_gpu_timed(
    const uint *h_image,
    const int *h_kernel,
    uint *h_output,
    int M,
    int N,
    int normalize_divisor,
    float *out_total_sec,
    float *out_kernel_ms
) {
    size_t img_bytes = sizeof(uint) * (size_t)M * M;
    size_t ker_bytes = sizeof(int) * (size_t)N * N;

    unsigned int *d_image = nullptr;
    int *d_kernel = nullptr;
    unsigned int *d_output = nullptr;


    cudaMalloc(&d_image, img_bytes);
    cudaMalloc(&d_kernel, ker_bytes);
    cudaMalloc(&d_output, img_bytes);

    clock_t t0 = clock();
    cudaMemcpy(d_image, h_image, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, ker_bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid( (M + TILE - 1) / TILE, (M + TILE - 1) / TILE );
    int S = TILE + 2*(N/2);
    size_t shared_bytes = sizeof(int) * S * S;
    conv2d_gpu_kernel<<<grid, block, shared_bytes>>>(d_image, d_kernel, d_output, M, N, normalize_divisor);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, img_bytes, cudaMemcpyDeviceToHost);
    clock_t t1 = clock();
    double total_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
    if (out_total_sec) *out_total_sec = (float)total_sec;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_image, h_image, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, ker_bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    conv2d_gpu_kernel<<<grid, block, shared_bytes>>>(d_image, d_kernel, d_output, M, N, normalize_divisor);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    if (out_kernel_ms) *out_kernel_ms = ms;


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}


int main(int argc, char **argv) {
    srand(12345);

    int Ms[3] = {512, 1024, 2048};
    int Ns[3] = {3, 5, 7};

    printf("M,N,filter,cpu_sec,gpu_total_sec,gpu_kernel_ms\n");

    for (int mi = 0; mi < 3; mi++) {
        int M = Ms[mi];
        uint *image = (uint*)malloc(sizeof(uint) * (size_t)M * M);
        uint *out_cpu = (uint*)malloc(sizeof(uint) * (size_t)M * M);
        uint *out_gpu = (uint*)malloc(sizeof(uint) * (size_t)M * M);

        generate_random_image(image, M);

        for (int ni = 0; ni < 3; ni++) {
            int N = Ns[ni];


            int *kernel = (int*)malloc(sizeof(int) * N * N);
            for (int k = 0; k < N*N; k++) kernel[k] = 1;
            int divisor = N*N;
            char fname[256];


            clock_t t0 = clock();
            conv2d_cpu_uint(image, kernel, out_cpu, M, N, divisor);
            clock_t t1 = clock();
            double cpu_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;


            float gpu_total = 0.0f;
            float gpu_kernel_ms = 0.0f;
            conv2d_gpu_timed(image, kernel, out_gpu, M, N, divisor, &gpu_total, &gpu_kernel_ms);


            printf("%d,%d,blur%dx%d,%.6f,%.6f,%.3f\n", M, N, N, N, cpu_sec, gpu_total, gpu_kernel_ms);


            snprintf(fname, sizeof(fname), "cpu_M%d_N%d_blur.pgm", M, N);
            save_pgm(fname, out_cpu, M);
            snprintf(fname, sizeof(fname), "gpu_M%d_N%d_blur.pgm", M, N);
            save_pgm(fname, out_gpu, M);

            free(kernel);
        }


        {
            int N = 3;
            int sobelx[9] = {-1,0,1, -2,0,2, -1,0,1};
            int lap3[9] = {0,-1,0, -1,4,-1, 0,-1,0};


            {
                int *k = (int*)malloc(sizeof(int)*9);
                memcpy(k, sobelx, sizeof(int)*9);
                // cpu
                clock_t t0 = clock();
                conv2d_cpu_uint(image, k, out_cpu, M, N, 1);
                clock_t t1 = clock();
                double cpu_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
                float gpu_total=0, gpu_kernel_ms=0;
                conv2d_gpu_timed(image, k, out_gpu, M, N, 1, &gpu_total, &gpu_kernel_ms);
                printf("%d,%d,sobelx,%.6f,%.6f,%.3f\n", M, N, cpu_sec, gpu_total, gpu_kernel_ms);
                char fname[256];
                snprintf(fname, sizeof(fname), "cpu_M%d_N3_sobelx.pgm", M);
                save_pgm(fname, out_cpu, M);
                snprintf(fname, sizeof(fname), "gpu_M%d_N3_sobelx.pgm", M);
                save_pgm(fname, out_gpu, M);
                free(k);
            }
            // laplacian
            {
                int *k = (int*)malloc(sizeof(int)*9);
                memcpy(k, lap3, sizeof(int)*9);
                clock_t t0 = clock();
                conv2d_cpu_uint(image, k, out_cpu, M, N, 1);
                clock_t t1 = clock();
                double cpu_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
                float gpu_total=0, gpu_kernel_ms=0;
                conv2d_gpu_timed(image, k, out_gpu, M, N, 1, &gpu_total, &gpu_kernel_ms);
                printf("%d,%d,laplacian,%.6f,%.6f,%.3f\n", M, N, cpu_sec, gpu_total, gpu_kernel_ms);
                char fname[256];
                snprintf(fname, sizeof(fname), "cpu_M%d_N3_laplacian.pgm", M);
                save_pgm(fname, out_cpu, M);
                snprintf(fname, sizeof(fname), "gpu_M%d_N3_laplacian.pgm", M);
                save_pgm(fname, out_gpu, M);
                free(k);
            }
        }

        free(image);
        free(out_cpu);
        free(out_gpu);
    }

    printf("All done.\n");
    return 0;
}
