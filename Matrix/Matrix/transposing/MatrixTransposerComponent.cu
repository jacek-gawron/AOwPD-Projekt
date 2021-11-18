#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MatrixTransposerComponent.h"
#include <stdexcept>
#include <thread>
#include <vector>

__global__ void transpose_GPU(float* a, float* out, size_t dim_x,
    size_t dim_y) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i < dim_x && j < dim_y) {
        if (dim_x == dim_y) {
            out[j*dim_y+i] = a[i*dim_y+j];
        }
        else if (dim_x < dim_y) {
            //z³y wzór (czasem dzia³a, czasem nie) (j*dim_y+1 nadaje siê tlyko do macierzy kwadratowych)
            out[j*dim_y+i-j] = a[i*dim_y+j];
        }
        else if (dim_x > dim_y) {
            //z³y wzór (czasem dzia³a, czasem nie) (j*dim_y+1 nadaje siê tlyko do macierzy kwadratowych)
            out[j*dim_y+i+j] = a[i*dim_y+j];
        }
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
    int num_of_bytes =
        output.get_x_dimension() * output.get_y_dimension() * sizeof(float);
    cudaMalloc((void**)&a_GPU_pointer, num_of_bytes);
    cudaMalloc((void**)&out_GPU_pointer, num_of_bytes);
    cudaMemcpy(a_GPU_pointer, a[0], num_of_bytes, cudaMemcpyHostToDevice);
    dim3 blocks(output.get_x_dimension(), output.get_y_dimension());
    transpose_GPU <<<blocks, 1 >>> (
        a_GPU_pointer, out_GPU_pointer,
        output.get_x_dimension(), output.get_y_dimension());
    cudaMemcpy(output[0], out_GPU_pointer, num_of_bytes, cudaMemcpyDeviceToHost);
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

