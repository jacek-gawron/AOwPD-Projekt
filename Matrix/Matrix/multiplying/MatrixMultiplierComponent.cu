#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MatrixMultiplierComponent.h"
#include <stdexcept>
#include <thread>
#include <vector>

__global__ void multiply_GPU(float* a, float* b, float* c,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < K && row < M) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * K + col];
        }
        c[row * K + col] = sum;
    }
}

void MatrixMultiplierComponent::set_matrices(Matrix a, Matrix b) {
    printf("%d %d", a.get_y_dimension(), b.get_x_dimension());
    if (a.get_x_dimension() != b.get_y_dimension()) {
        throw std::invalid_argument("These matrices cannot be mutiplied");
    }
    this->a = a;
    this->b = b;
    this->output = Matrix(a.get_matrix_name() + b.get_matrix_name() + "multiplyOP",
        b.get_x_dimension(), a.get_y_dimension());
}

void MatrixMultiplierComponent::multiply_matrices_CPU_single_thread() {
    int r1 = output.get_y_dimension();
    int r2 = b.get_y_dimension();
    int c2 = output.get_x_dimension();

    for (int i{ 0 }; i < r1; i++) {
        for (int j{ 0 }; j < c2; j++) {
            output[i][j] = 0;
            for (int k{ 0 }; k < r2; k++) {
                output[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void MatrixMultiplierComponent::multiply_matrices_CPU_multi_thread() {
    num_of_cores = output.get_x_dimension();
    std::vector<std::thread> threads;
    for (int i{ 0 }; i < num_of_cores; i++) {
        threads.push_back(std::thread(&MatrixMultiplierComponent::thread_CPU_fun, this, i));
    }

    for (int i{ 0 }; i < num_of_cores; i++) {
        threads[i].join();
    }
}

void MatrixMultiplierComponent::multiply_matrices_GPU() {
    float* a_GPU_pointer, * b_GPU_pointer, * out_GPU_pointer;
    int a_bytes =
        a.get_x_dimension() * a.get_y_dimension() * sizeof(float);
    int b_bytes =
        b.get_x_dimension() * b.get_y_dimension() * sizeof(float);
    int out_bytes =
        output.get_x_dimension() * output.get_y_dimension() * sizeof(float);
    cudaMalloc((void**)&a_GPU_pointer, a_bytes);
    cudaMalloc((void**)&b_GPU_pointer, b_bytes);
    cudaMalloc((void**)&out_GPU_pointer, out_bytes);

    cudaMemcpy(a_GPU_pointer, a[0], a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_GPU_pointer, b[0], b_bytes, cudaMemcpyHostToDevice);
    int BLOCK_SIZE = 16;
    unsigned int grid_rows = (a.get_x_dimension() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (b.get_y_dimension() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocksPerGrid(grid_cols, grid_rows);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    multiply_GPU<<<blocksPerGrid, threadsPerBlock>>> (
        a_GPU_pointer, b_GPU_pointer, out_GPU_pointer,
        a.get_y_dimension(), a.get_x_dimension(), b.get_x_dimension());
    cudaMemcpy(output[0], out_GPU_pointer, out_bytes, cudaMemcpyDeviceToHost);
    cudaFree(a_GPU_pointer);
    cudaFree(b_GPU_pointer);
    cudaFree(out_GPU_pointer);
}

Matrix MatrixMultiplierComponent::get_result() { return output; }

void MatrixMultiplierComponent::thread_CPU_fun(int thread_id) {
    int r2 = b.get_y_dimension();
    int c2 = output.get_x_dimension();
    for (int j{ 0 }; j < c2; j++) {
        output[thread_id][j] = 0;
        for (int k{ 0 }; k < r2; k++) {
            this->output[thread_id][j] += a[thread_id][k] * b[k][j];
        }
    }
}

