#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MatrixMultiplierComponent.h"
#include <stdexcept>
#include <thread>
#include <vector>

void MatrixMultiplierComponent::set_matrices(Matrix a, Matrix b) {
    if (a.get_y_dimension() != b.get_x_dimension()) {
        throw std::invalid_argument("These matrices cannot bve mutiplied");
    }
    this->a = a;
    this->b = b;
    this->output = Matrix(a.get_matrix_name() + b.get_matrix_name() + "multiplyOP",
        a.get_y_dimension(), b.get_x_dimension());
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

__global__ void multiply_GPU(float* a, float* b, float* out, size_t dim_x, size_t dim_y) {

}
