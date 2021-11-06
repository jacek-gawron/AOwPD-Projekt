#include "MatrixTransposerComponent.h"
#include <stdexcept>
#include <thread>
#include <vector>

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

