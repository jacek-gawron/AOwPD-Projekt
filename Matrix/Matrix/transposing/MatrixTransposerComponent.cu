#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MatrixTransposerComponent.h"
#include <stdexcept>
#include <thread>
#include <vector>

__global__ void transpose_GPU(float* a, float* out, int dim_x,
    int dim_y) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < dim_y && idy < dim_x)
    {
        unsigned int pos = idy * dim_y + idx;
        unsigned int newPos = idx * dim_x + idy;
        out[newPos] = a[pos];
    }
}

void MatrixTransposerComponent::set_matrix(Matrix a) {
    this->a = a;
    this->output = Matrix(a.get_matrix_name() + "transposeOP",
        a.get_y_dimension(), a.get_x_dimension());
}

void MatrixTransposerComponent::transpose_matrix_CPU_single_thread() {
    int max_x = output.get_x_dimension(), max_y = output.get_y_dimension();
    for (int i{ 0 }; i < max_y; i++) {
        for (int j{ 0 }; j < max_x; j++) {
            output[i][j] = a[j][i];
        }
    }
}

void MatrixTransposerComponent::transpose_matrix_CPU_multi_thread() {
    num_of_cores = std::thread::hardware_concurrency();
    if (!num_of_cores) num_of_cores = output.get_y_dimension();
    std::vector<std::thread> threads;

    for (int i{ 0 }; i < num_of_cores; i++) {
        threads.push_back(std::thread(&MatrixTransposerComponent::thread_CPU_fun, this, i));
    }

    for (int i{ 0 }; i < num_of_cores; i++) {
        threads[i].join();
    }
}

void MatrixTransposerComponent::transpose_matrix_GPU() {
    float *a_GPU_pointer, *out_GPU_pointer;
    int a_bytes =
        a.get_x_dimension() * a.get_y_dimension() * sizeof(float);
    int out_bytes =
        output.get_x_dimension() * output.get_y_dimension() * sizeof(float);
    cudaMalloc((void**)&a_GPU_pointer, a_bytes);
    cudaMalloc((void**)&out_GPU_pointer, out_bytes);
    cudaMemcpy(a_GPU_pointer, a[0], a_bytes, cudaMemcpyHostToDevice);
    int BLOCK_SIZE = 16;
    unsigned int grid_rows = (a.get_y_dimension() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (a.get_y_dimension() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocksPerGrid(grid_cols, grid_rows);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    transpose_GPU << <blocksPerGrid, threadsPerBlock >> > (
        a_GPU_pointer, out_GPU_pointer,
        a.get_y_dimension(), a.get_x_dimension());
    cudaMemcpy(output[0], out_GPU_pointer, out_bytes, cudaMemcpyDeviceToHost);
    cudaFree(a_GPU_pointer);
    cudaFree(out_GPU_pointer);
}

Matrix MatrixTransposerComponent::get_result() { return output; }

void MatrixTransposerComponent::thread_CPU_fun(int thread_id) {
    int max_x = output.get_x_dimension(), max_y = output.get_y_dimension();
    for (int i{ thread_id }; i < max_y; i += num_of_cores) {
        for (int j{ 0 }; j < max_x; j++) {
            output[i][j] = a[j][i];
        }
    }
}

