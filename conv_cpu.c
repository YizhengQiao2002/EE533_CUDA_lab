// conv_cpu.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

typedef unsigned int uint;

void conv2d_cpu_uint(
    const uint *image,    // M x M
    const int *kernel,    // N x N 
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
    for (int i = 0; i < M * M; i++) {
        img[i] = rand() % 256;
    }
}

void save_pgm(const char *filename, const uint *img, int M) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Unable to open %s for writing\n", filename);
        return;
    }
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

void fill_kernels(int **kernels, int *sizes, char ***names, int *count) {
    *count = 3;
    *kernels = (int*)malloc(sizeof(int) * 3 * 7 * 7); 
    *sizes = 3; 
    *names = (char**)malloc(sizeof(char*) * 3);

   
    int blur3[9] = {1,1,1, 1,1,1, 1,1,1};
    memcpy((*kernels) + 0*49, blur3, sizeof(int)*9);
    (*names)[0] = strdup("blur3");

    int sobelx[9] = {-1,0,1, -2,0,2, -1,0,1};
    memcpy((*kernels) + 1*49, sobelx, sizeof(int)*9);
    (*names)[1] = strdup("sobelx3");

    int lap3[9] = {0,-1,0, -1,4,-1, 0,-1,0};
    memcpy((*kernels) + 2*49, lap3, sizeof(int)*9);
    (*names)[2] = strdup("lap3");
}

int main(int argc, char **argv) {
    srand(12345);

    int Ms[3] = {512, 1024, 2048};      // image sizes
    int Ns[3] = {3, 5, 7};            


    for (int mi = 0; mi < 3; mi++) {
        int M = Ms[mi];

        uint *image = (uint*)malloc(sizeof(uint) * M * M);
        uint *output = (uint*)malloc(sizeof(uint) * M * M);

        generate_random_image(image, M);

        for (int ni = 0; ni < 3; ni++) {
            int N = Ns[ni];

            int *kernel = (int*)malloc(sizeof(int) * N * N);
            int divisor = 1;
            char filtername[64];

            
            for (int k = 0; k < N*N; k++) kernel[k] = 1;
            divisor = N*N;
            snprintf(filtername, sizeof(filtername), "blur%dx%d", N, N);

      
            clock_t t0 = clock();
            conv2d_cpu_uint(image, kernel, output, M, N, divisor);
            clock_t t1 = clock();
            double tsec = (double)(t1 - t0) / CLOCKS_PER_SEC;
            printf("M=%d N=%d filter=%s time=%.4f sec\n", M, N, filtername, tsec);


            char fname[256];
            snprintf(fname, sizeof(fname), "out_M%d_N%d_%s.pgm", M, N, filtername);
            save_pgm(fname, output, M);

            free(kernel);

       
            if (N == 3) {
                int sobelx[9] = {-1,0,1, -2,0,2, -1,0,1};
                int lap3[9] = {0,-1,0, -1,4,-1, 0,-1,0};
                int *k2 = (int*)malloc(sizeof(int)*9);
                memcpy(k2, sobelx, sizeof(int)*9);
                t0 = clock();
                conv2d_cpu_uint(image, k2, output, M, 3, 1);
                t1 = clock();
                tsec = (double)(t1 - t0) / CLOCKS_PER_SEC;
                printf("M=%d N=3 filter=sobelx time=%.4f sec\n", M, tsec);
                snprintf(fname, sizeof(fname), "out_M%d_N3_sobelx.pgm", M);
                save_pgm(fname, output, M);
                free(k2);

                int *k3 = (int*)malloc(sizeof(int)*9);
                memcpy(k3, lap3, sizeof(int)*9);
                t0 = clock();
                conv2d_cpu_uint(image, k3, output, M, 3, 1);
                t1 = clock();
                tsec = (double)(t1 - t0) / CLOCKS_PER_SEC;
                printf("M=%d N=3 filter=laplacian time=%.4f sec\n", M, tsec);
                snprintf(fname, sizeof(fname), "out_M%d_N3_laplacian.pgm", M);
                save_pgm(fname, output, M);
                free(k3);
            }
        }

        free(image);
        free(output);
    }

    printf("All done.\n");
    return 0;
}
