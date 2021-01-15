/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include csrms2.cpp 
 *   g++ -o csrm2 csrsm2.o -L/usr/local/cuda/lib64 -lcusparse -lcudart
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

void init_matrix_A1(Matrix *A){
    printf("init_matrix_A1()\n");
    A->height = 3;
    A->width = 3;
    int element_count = A->width*A->height;
    A->elements = (float*)malloc(sizeof(float)*9);
    float elements[9] = {1,0,3,0,0,6,7,0,9};
    for(int i = 0; i < element_count; i++){
        A->elements[i] = elements[i];
    }
}

void print_dense_matrix_elements(Matrix *A){
    int element_count = A->width*A->height;
    for(int i = 0; i < element_count; i++){
        printf("%f\n", A->elements[i]);
    }
}

int main(){
    printf("main()\n");
    Matrix a1 = Matrix();
    init_matrix_A1(&a1);
    print_dense_matrix_elements(&a1);
    return 0;
}