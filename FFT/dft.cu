#include <stdio.h>
#include <cuComplex.h>
#include <complex.h>
#include <math.h>

#define M_PI 3.14159265358979323846
#define ARRAY_SIZE 1000000
#define TPB 256

typedef float2 Cplx;

__global__ void DFT(float *arr, Cplx *DFTArr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ARRAY_SIZE) {
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            DFTArr[idx].x += arr[i] * cos((2 * M_PI * idx * i) / ARRAY_SIZE);
            DFTArr[idx].y -= arr[i] * sin((2 * M_PI * idx * i) / ARRAY_SIZE);
        }
    }
}

int main(int argc, char **argv) {
    
    float *arr = (float*)malloc(ARRAY_SIZE * sizeof(arr[0]));
    float *d_arr = (float*)malloc(ARRAY_SIZE * sizeof(arr[0]));
    Cplx *DFTArr = (Cplx*)malloc(ARRAY_SIZE * sizeof(DFTArr[0]));
    Cplx *d_DFTArr = (Cplx*)malloc(ARRAY_SIZE * sizeof(DFTArr[0]));
    cudaMalloc(&d_DFTArr, ARRAY_SIZE * sizeof(DFTArr[0]));
    cudaMalloc(&d_arr, ARRAY_SIZE * sizeof(arr[0]));
    for(int i = 0; i < ARRAY_SIZE; ++i) {
        arr[i] = i % 5;
    }
    cudaMemcpy(d_arr, arr, ARRAY_SIZE * sizeof(d_arr[0]), cudaMemcpyHostToDevice);
    DFT<<<(ARRAY_SIZE + TPB - 1)/TPB, TPB>>>(d_arr, d_DFTArr);
    cudaMemcpy(DFTArr, d_DFTArr, ARRAY_SIZE * sizeof(DFTArr[0]), cudaMemcpyDeviceToHost);

    for (int j = 0; j < 30; j++) {
        printf("The real part of %d element is %f\n", j, DFTArr[j].x);
        printf("The imaginary part of %d element is %f\n\n", j, DFTArr[j].y);
    }
    cudaFree(d_arr); cudaFree(d_DFTArr);
    free(arr); free(DFTArr);
    return 0;
}