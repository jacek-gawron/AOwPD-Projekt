#pragma once
#include "../utils/matrix/Matrix.hpp"

class MatrixTransposerComponent {
public:
    void set_matrix(Matrix a);

    void transpose_matrix_CPU_single_thread();
    void transpose_matrix_CPU_multi_thread();
    void transpose_matrix_GPU();
    unsigned int get_num_of_threads() { return num_of_cores; }

    Matrix get_result();

private:
    Matrix a, output;
    unsigned int num_of_cores = 0;

    void thread_CPU_fun(int thread_id);
};