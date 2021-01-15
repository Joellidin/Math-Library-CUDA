#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#define max(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b;       \
})

#define min(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b;       \
})

#define A1ROWS 10
#define A1COLUMNS 10
#define N 200

/*
COMPILE:
nvcc matrix_vector_gpu.cu -arch=sm_75 -o matrix_vector_gpu 
*/

template <typename data_type>
__global__ void coo_spmv_kernel (
    unsigned int n_elements,
    const unsigned int *col_ids,
    const unsigned int *row_ids,
    const data_type *data,
    const data_type *x,
    data_type *y
){
    unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
    //y[element] = 1.0;
    if(element < n_elements){
        atomicAdd(y + row_ids[element], data[element] * x[col_ids[element]]);
    }
}

/*
nnz 	(integer) 	The number of nonzero elements in the matrix.
cooValA 	(pointer) 	Points to the data array of length nnz that holds all nonzero values of A in row-major format.
cooRowIndA 	(pointer) 	Points to the integer array of length nnz that contains the row indices of the corresponding elements in array cooValA.
cooColIndA 	(pointer) 	Points to the integer array of length nnz that contains the column indices of the corresponding elements in array cooValA. 
*/


typedef struct {
    unsigned int nnz;
    float *cooValA;
    unsigned int *cooRowIndA;
    unsigned int *cooColIndA;
} coo_matrix;


typedef struct {
    unsigned int width;
    unsigned int height;
    float *elements;
    unsigned int element_count;
} dense_matrix;

void convert_dense_to_coo(dense_matrix* matrix_a, coo_matrix *result_matrix){
    unsigned int nnz = 0;
    for(unsigned int row_index = 0; row_index < matrix_a->height; row_index++){
        for(unsigned int col_index = 0; col_index < matrix_a->width; col_index++){
            unsigned int element_index = row_index*matrix_a->width + col_index;
            float value = matrix_a->elements[element_index];
            if(value != 0.0){
                nnz++;
            }
        }
    }
    /*
    float *cooValA = (float*)calloc(nnz, sizeof(float));
    unsigned int *cooRowIndA = (unsigned int*)calloc(nnz, sizeof(unsigned int));
    unsigned int *cooColIndA = (unsigned int*)calloc(nnz, sizeof(unsigned int));
    */
    float *cooValA;
    cudaMallocManaged(&cooValA, nnz * sizeof(float));
    unsigned int *cooRowIndA;
    cudaMallocManaged(&cooRowIndA, nnz * sizeof(unsigned int));
    unsigned int *cooColIndA;
    cudaMallocManaged(&cooColIndA, nnz * sizeof(unsigned int));
    unsigned int cooValAIndex = 0;
    for(unsigned int row_index = 0; row_index < matrix_a->height; row_index++){
        for(unsigned int col_index = 0; col_index < matrix_a->width; col_index++){
            unsigned int element_index = row_index*matrix_a->width + col_index;
            float value = matrix_a->elements[element_index];
            if(value != 0.0){
                //printf("value %f %ld %ld\n", value, row_index, col_index);
                cooValA[cooValAIndex] = value;
                cooRowIndA[cooValAIndex] = row_index;
                cooColIndA[cooValAIndex] = col_index;
                cooValAIndex++;
            }
        }
    }
    result_matrix->nnz = nnz;
    result_matrix->cooValA = cooValA;
    result_matrix->cooRowIndA = cooRowIndA;
    result_matrix->cooColIndA = cooColIndA;
}

void fill_dense_matrix_diagonal_single_value(dense_matrix *matrix_a, float single_value){
    unsigned int shortest_side = min(matrix_a->height, matrix_a->width);
    for(unsigned int i = 0; i < shortest_side; i++){
        unsigned int element_index = i * matrix_a->width + i;
        matrix_a->elements[element_index] = single_value;
    }
}

void fill_band_matrix_single_value(dense_matrix *matrix_a, float single_value, int band_width){
    unsigned int shortest_side = min(matrix_a->height, matrix_a->width);
    for(int i = 0; i < shortest_side; i++){
        int diagonal_index = i * matrix_a->width + i;
        matrix_a->elements[diagonal_index] = single_value;
        if(i != (shortest_side-1)){
            int below_index = (i+1) * matrix_a->width + i;
            int right_index = i * matrix_a->width + (i+1);
            matrix_a->elements[below_index] = single_value;
            matrix_a->elements[right_index] = single_value;
        }
    }
}

void fill_dense_matrix_interval_single_value(dense_matrix* matrix_a, float single_value, int interval){
    for(unsigned int i = 0; i < matrix_a->element_count; i = i + interval){
        matrix_a->elements[i] = single_value;
    }
}

dense_matrix* get_matrix_a1(){
    dense_matrix* matrix_a1 = (dense_matrix*)malloc(sizeof(dense_matrix));
    matrix_a1->height = A1ROWS;
    matrix_a1->width = A1COLUMNS;
    matrix_a1->element_count = matrix_a1->height * matrix_a1->width;
    matrix_a1->elements = (float*)calloc(sizeof(float), matrix_a1->element_count);
    return matrix_a1;
}

dense_matrix* get_vector_b1(){
    dense_matrix* matrix_a1 = (dense_matrix*)malloc(sizeof(dense_matrix));
    matrix_a1->height = A1COLUMNS;
    matrix_a1->width = 1;
    matrix_a1->element_count = matrix_a1->height * matrix_a1->width;
    //matrix_a1->elements = (float*)calloc(sizeof(float), matrix_a1->element_count);
    cudaMallocManaged(&matrix_a1->elements, matrix_a1->element_count * sizeof(float));
    return matrix_a1;
}

dense_matrix *init_result_vector(){
    dense_matrix* vector_c = (dense_matrix*)malloc(sizeof(dense_matrix));
    vector_c->height = A1COLUMNS;
    vector_c->width = 1;
    vector_c->element_count = vector_c->height * vector_c->width;
    //vector_c->elements = (float*)calloc(sizeof(float), vector_c->element_count);
    cudaMallocManaged(&vector_c->elements, vector_c->element_count * sizeof(float));
    return vector_c;
}

void print_dense_matrix(dense_matrix *matrix_a){
    printf("[\n");
    for(unsigned int row_index = 0; row_index < matrix_a->height; row_index++){
        //printf("row %d", row_index);
        printf("[");
        for(unsigned int col_index = 0; col_index < matrix_a->width; col_index++){
            unsigned int element_index = row_index * matrix_a->width + col_index;
            if(col_index == (matrix_a->width - 1)){
                printf("%f", matrix_a->elements[element_index]);
            } else {
                printf("%f,", matrix_a->elements[element_index]);
            }
        }
        printf("],");
        printf("\n");
    }
    printf("]\n");
    return;
}

void print_coo_matrix(coo_matrix *matrix_a){
    printf("cooValA: ");
    for(unsigned int i = 0; i < matrix_a->nnz; i++){
        printf("%f ", matrix_a->cooValA[i]);
    }
    printf("\n");
    printf("cooRowIndA: ");
    for(unsigned int i = 0; i < matrix_a->nnz; i++){
        printf("%ld ", matrix_a->cooRowIndA[i]);
    }
    printf("\n");
    printf("cooRowColA: ");
    for(unsigned int i = 0; i < matrix_a->nnz; i++){
        printf("%ld ", matrix_a->cooColIndA[i]);
    }
}

int main(){
    printf("main()\n");
    dense_matrix *matrix_a1 = get_matrix_a1();
    dense_matrix *vector_b1 = get_vector_b1();
    dense_matrix *result_vector = init_result_vector();
    fill_dense_matrix_diagonal_single_value(matrix_a1, 1.0);
    fill_band_matrix_single_value(matrix_a1, 3.0, 1);
    fill_dense_matrix_interval_single_value(vector_b1, 1.0, 3);
    print_dense_matrix(matrix_a1);
    print_dense_matrix(vector_b1);
    coo_matrix *matrix_a1_coo = (coo_matrix*)malloc(sizeof(coo_matrix));
    convert_dense_to_coo(matrix_a1, matrix_a1_coo);
    //print_coo_matrix(matrix_a1_coo);
    int numBlocks = 1;
    //dim3 threadsPerBlock(N, N);
    coo_spmv_kernel<<<numBlocks, 32*10>>>(matrix_a1_coo->nnz,
                                                    matrix_a1_coo->cooColIndA,
                                                    matrix_a1_coo->cooRowIndA,
                                                    matrix_a1_coo->cooValA,
                                                    vector_b1->elements,
                                                    result_vector->elements);
    cudaDeviceSynchronize();
    printf("after kernel\n");
    print_dense_matrix(result_vector);
    return 0;
}