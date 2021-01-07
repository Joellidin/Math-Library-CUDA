#include <stdio.h>

// THIS IS BASED ON https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction
// 3.2.4 Shared Memory, without stride field

#define N 2
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 2

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

void fill_matrix_with_values(Matrix A, float value, const int element_count){
    for(int i = 0; i < element_count; i++){
        A.elements[i] = value;
    }
}

int main(){
    printf("Hello I am main.cu\n");
    const int WIDTH = BLOCK_SIZE;
    const int HEIGHT = BLOCK_SIZE;
    const int ELEMENT_COUNT = WIDTH*HEIGHT;
    Matrix A = {.width = WIDTH, .height = HEIGHT, .elements = (float*)calloc(ELEMENT_COUNT, sizeof(float)) };
    Matrix B = {.width = WIDTH, .height = HEIGHT, .elements = (float*)calloc(ELEMENT_COUNT, sizeof(float)) };
    Matrix C = {.width = WIDTH, .height = HEIGHT, .elements = (float*)calloc(ELEMENT_COUNT, sizeof(float)) };
    fill_matrix_with_values(A, 1.0, ELEMENT_COUNT);
    fill_matrix_with_values(B, 2.0, ELEMENT_COUNT);
    fill_matrix_with_values(C, 0.0, ELEMENT_COUNT);
    printf("A.elements[0] %f \n", A.elements[0]);
    printf("Before C.elements[0] %f \n", C.elements[0]);
    MatMul(A, B, C);
    cudaDeviceSynchronize();
    printf("Matrix C \n");
    printf("C.elements[0] %f \n", C.elements[0]);
    printf("C.elements[ELEMENT_COUNT-1] %f \n", C.elements[ELEMENT_COUNT-1]);
    return 0;
}