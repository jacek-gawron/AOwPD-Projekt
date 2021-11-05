#pragma once
#include "../utils/matrix/Matrix.hpp"

class MatrixAdderComponent {
 public:
  void set_matrices(Matrix a, Matrix b);

  void add_matrices_CPU_single_thread();
  void add_matrices_CPU_multi_thread();
  void add_matrices_GPU();
  unsigned int get_num_of_threads() { return num_of_cores; }

  Matrix get_result();

 private:
  Matrix a, b, output;
  unsigned int num_of_cores = 0;

  void thread_CPU_fun(int thread_id);
};
