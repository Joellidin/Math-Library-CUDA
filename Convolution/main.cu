#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <errno.h>

#define BLOCK_SIZE  16
#define BLOCK_SIZE_SH 18
#define HEADER_SIZE 122
#define FILTER_DIM 3

typedef unsigned char BYTE;

/**
 * Structure that represents a BMP image.
 */
typedef struct
{
    int   width;
    int   height;
    float *data;
} BMPImage;

typedef struct timeval tval;

BYTE g_info[HEADER_SIZE]; // Reference header

/**
 * Reads a BMP 24bpp file and returns a BMPImage structure.
 * Thanks to https://stackoverflow.com/a/9296467
 */
BMPImage readBMP(char *filename)
{
    BMPImage bitmap = { 0 };
    int      size   = 0;
    BYTE     *data  = NULL;
    FILE     *file  = fopen(filename, "rb");
    
    // Read the header (expected BGR - 24bpp)
    fread(g_info, sizeof(BYTE), HEADER_SIZE, file);

    // Get the image width / height from the header
    bitmap.width  = *((int *)&g_info[18]);
    bitmap.height = *((int *)&g_info[22]);
    size          = *((int *)&g_info[34]);
    
    // Read the image data
    data = (BYTE *)malloc(sizeof(BYTE) * size);
    fread(data, sizeof(BYTE), size, file);
    
    // Convert the pixel values to float
    bitmap.data = (float *)malloc(sizeof(float) * size);
    
    for (int i = 0; i < size; i++)
    {
        bitmap.data[i] = (float)data[i];
    }
    
    fclose(file);
    free(data);
    
    return bitmap;
}


/**
 * Writes a BMP file in grayscale given its image data and a filename.
 */
void writeBMPGrayscale(int width, int height, float *image, char *filename)
{
    FILE *file = NULL;
    
    file = fopen(filename, "wb");
    
    // Write the reference header
    fwrite(g_info, sizeof(BYTE), HEADER_SIZE, file);
    
    // Unwrap the 8-bit grayscale into a 24bpp (for simplicity)
    for (int h = 0; h < height; h++)
    {
        int offset = h * width;
        
        for (int w = 0; w < width; w++)
        {
            BYTE pixel = (BYTE)((image[offset + w] > 255.0f) ? 255.0f :
                                (image[offset + w] < 0.0f)   ? 0.0f   :
                                                               image[offset + w]);
            
            // Repeat the same pixel value for BGR
            fputc(pixel, file);
            fputc(pixel, file);
            fputc(pixel, file);
        }
    }
    
    fclose(file);
}

void store_result(int index, double elapsed_gpu,
                     int width, int height, float *image)
{
    char path[255];
    
    sprintf(path, "Images/result_%d.bmp", index);
    printf("%s", path);
    writeBMPGrayscale(width, height, image, path);
    
    printf("Completed - Result stored in \"%s\".\n", path);
    printf("Elapsed GPU: %fms\n", elapsed_gpu);
}


/**
 * Releases a given BMPImage.
 */
void freeBMP(BMPImage bitmap)
{
    free(bitmap.data);
}

__global__ void gpu_grayscale(int width, int height, float *image, float *image_out)
{
    ////////////////
    // TO-DO #4.2 /////////////////////////////////////////////
    // Implement the GPU version of the grayscale conversion //
    ///////////////////////////////////////////////////////////
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < width && idy < height) {
        int offset = idy * width + idx;
        image_out[offset] = image[3 * offset] * 0.0722f + // B
                            image[3 * offset + 1] * 0.7152f + // G
                            image[3 * offset + 2] * 0.2126f;  // R
    } 
}

__device__ float gpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
    float pixel = 0.0f;
    
    for (int h = 0; h < filter_dim; h++)
    {
        int offset        = h * stride;
        int offset_kernel = h * filter_dim;
        
        for (int w = 0; w < filter_dim; w++)
        {
            pixel += image[offset + w] * matrix[offset_kernel + w];
        }
    }
    
    return pixel;
}

__global__ void gpu_convolve(int width, int height, float *image, float *image_out, float *filter, int filter_dim) {
    __shared__ float sh_block[BLOCK_SIZE_SH * BLOCK_SIZE_SH];
    // float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
    //                       2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
    //                       1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (index_x < (width - 2) && index_y < (height - 2))  {
        int shared_id = threadIdx.y * BLOCK_SIZE_SH + threadIdx.x;
        sh_block[shared_id] = image[index_y * width + index_x];
        if (threadIdx.x >= BLOCK_SIZE - 2) {
            sh_block[shared_id + 2] = image[index_y * width + index_x + 2];
        }
        if (threadIdx.y >= BLOCK_SIZE - 2) {
            int id = (threadIdx.y + 2) * BLOCK_SIZE_SH + threadIdx.x;
            sh_block[id] = image[(index_y + 2) * width + index_x];
        }
        if (threadIdx.x >= BLOCK_SIZE - 2 && threadIdx.y >= BLOCK_SIZE - 2) {
            int id = (threadIdx.y + 2) * BLOCK_SIZE_SH + threadIdx.x + 2;
            sh_block[id] = image[(index_y + 2) * width + index_x + 2];
        }
        __syncthreads();

        int offset   = (index_y + 1) * width + (index_x + 1);
        image_out[offset] = gpu_applyFilter(&sh_block[shared_id],
                                            BLOCK_SIZE_SH, filter, filter_dim);
    }
}

double get_elapsed(tval t0, tval t1)
{
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}


int main(int argc, char **argv) {

    BMPImage bitmap          = { 0 };
    float    *d_bitmap       = { 0 };
    float    *image_out[2]   = { 0 };
    float    *d_image_out[2] = { 0 };
    float    *gpu_filter     = { 0 };
    int      image_size      = 0;
    tval     t[2]            = { 0 };
    double   elapsed      = 0;
    dim3     grid(1);                       // The grid will be defined later
    dim3     block(BLOCK_SIZE, BLOCK_SIZE); // The block size will not change
    
    // Make sure the filename is provided
    if (argc != 2)
    {
        fprintf(stderr, "Error: The filename is missing!\n");
        return -1;
    }
    
    // Read the input image and update the grid dimension
    bitmap     = readBMP(argv[1]);
    image_size = bitmap.width * bitmap.height;
    grid       = dim3(((bitmap.width  + (BLOCK_SIZE - 1)) / BLOCK_SIZE),
                      ((bitmap.height + (BLOCK_SIZE - 1)) / BLOCK_SIZE));
    
    printf("Image opened (width=%d height=%d).\n", bitmap.width, bitmap.height);
    
    // Allocate the intermediate image buffers for each step
    for (int i = 0; i < 2; i++)
    {
        image_out[i] = (float *)calloc(image_size, sizeof(float));
        
        cudaMalloc(&d_image_out[i], image_size * sizeof(float));
        cudaMemset(d_image_out[i], 0, image_size * sizeof(float));
    }

    cudaMalloc(&d_bitmap, image_size * sizeof(float) * 3);
    cudaMemcpy(d_bitmap, bitmap.data,
               image_size * sizeof(float) * 3, cudaMemcpyHostToDevice);

    // Step 1: Convert to grayscale
    {   
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        gpu_grayscale<<<grid, block>>>(bitmap.width, bitmap.height,
                                       d_bitmap, d_image_out[0]);
        cudaMemcpy(image_out[0], d_image_out[0],
                   image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        
        elapsed = get_elapsed(t[0], t[1]);
        // Store the result image in grayscale
        store_result(1, elapsed, bitmap.width, bitmap.height, image_out[0]);
    }

    float filter[FILTER_DIM * FILTER_DIM] = {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                              2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                              1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f};
    
    cudaMalloc(&gpu_filter, FILTER_DIM * FILTER_DIM * sizeof(float));   
    cudaMemcpy(gpu_filter, filter,
        FILTER_DIM * FILTER_DIM * sizeof(float), cudaMemcpyHostToDevice);
    // Launch the GPU version
    gettimeofday(&t[0], NULL);
    gpu_convolve<<<grid, block>>>(bitmap.width, bitmap.height,
                                d_image_out[0], d_image_out[1], gpu_filter, FILTER_DIM);
    cudaMemcpy(image_out[1], d_image_out[1],
            image_size * sizeof(float), cudaMemcpyDeviceToHost);
    gettimeofday(&t[1], NULL);
    
    elapsed = get_elapsed(t[0], t[1]);
    
    // Store the result image with the Gaussian filter applied
    store_result(2, elapsed, bitmap.width, bitmap.height, image_out[1]);

    // Release the allocated memory
    for (int i = 0; i < 2; i++)
    {
        free(image_out[i]);
        cudaFree(d_image_out[i]);
    }
    
    freeBMP(bitmap);
    cudaFree(d_bitmap);
    cudaFree(gpu_filter);

    return 0;
}