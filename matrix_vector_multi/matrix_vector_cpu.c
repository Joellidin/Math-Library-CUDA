
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

#define A1ROWS 4
#define A1COLUMNS 4


typedef struct {
    long width;
    long height;
    float *elements;
    long element_count;
} dense_matrix;

void compute_matrix_vector_multiplication(dense_matrix *matrix_a, dense_matrix *vector_b, dense_matrix *result){
    assert(matrix_a->width == vector_b->height);
    assert(vector_b->height == result->height);
    long common_dimension_size = matrix_a->width;
    for(long matrix_row_index = 0; matrix_row_index < matrix_a->height; matrix_row_index++){
        for(long i = 0; i < common_dimension_size; i++){
            long element_index = matrix_row_index * matrix_a->width + i;
            float product = matrix_a->elements[element_index] * vector_b->elements[i];
            result->elements[i] = result->elements[i] + product;
        }
    }
}

void print_dense_matrix(dense_matrix *matrix_a){
    printf("[\n");
    for(long row_index = 0; row_index < matrix_a->height; row_index++){
        //printf("row %d", row_index);
        printf("[");
        for(long col_index = 0; col_index < matrix_a->width; col_index++){
            long element_index = row_index * matrix_a->width + col_index;
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

void fill_dense_matrix_diagonal_single_value(dense_matrix *matrix_a, float single_value){
    long shortest_side = min(matrix_a->height, matrix_a->width);
    for(long i = 0; i < shortest_side; i++){
        long element_index = i * matrix_a->width + i;
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
    for(long i = 0; i < matrix_a->element_count; i = i + interval){
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
    matrix_a1->elements = (float*)calloc(sizeof(float), matrix_a1->element_count);
    return matrix_a1;
}

dense_matrix *init_result_vector(){
    dense_matrix* vector_c = (dense_matrix*)malloc(sizeof(dense_matrix));
    vector_c->height = A1COLUMNS;
    vector_c->width = 1;
    vector_c->element_count = vector_c->height * vector_c->width;
    vector_c->elements = (float*)calloc(sizeof(float), vector_c->element_count);
    return vector_c;
}

int main(){
    /*
    COMPILE WITH: gcc matrix_vector_cpu.c -o matrix_vector_cpu
    */
    struct timeval start_time;
    struct timeval end_time;
    gettimeofday(&start_time, NULL);
    printf("main()\n");
    dense_matrix* matrix_a1 = get_matrix_a1();
    dense_matrix* vector_b1 = get_vector_b1();
    printf("dense_matrix->element_count %d\n", matrix_a1->element_count);
    fill_dense_matrix_diagonal_single_value(matrix_a1, 1.0);
    fill_band_matrix_single_value(matrix_a1, 2.0, 1);
    fill_dense_matrix_interval_single_value(vector_b1, 1.0, 3);
    print_dense_matrix(matrix_a1);
    print_dense_matrix(vector_b1);
    dense_matrix *result_vector = init_result_vector();
    printf("Before compute_matrix_vector_multiplication()\n");
    print_dense_matrix(result_vector);
    compute_matrix_vector_multiplication(matrix_a1, vector_b1, result_vector);
    gettimeofday(&end_time, NULL);
    printf("Final result\n");
    print_dense_matrix(result_vector);
    free(matrix_a1->elements);
    free(matrix_a1);
    printf("Time taken to count to 10^5 is : %ld micro seconds\n",
    ((end_time.tv_sec * 1000000 + end_time.tv_usec) -
    (start_time.tv_sec * 1000000 + start_time.tv_usec)));
}