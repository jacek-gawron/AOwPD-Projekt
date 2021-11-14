#pragma once
#include "../utils/matrix/Matrix.hpp"

class MatrixInverseComponent {
 public:
  void set_matrix(Matrix a);

  unsigned int get_num_of_threads() { return num_of_cores; }

  void inverse_matrix_CPU_single_thread();
  void inverse_matrix_CPU_multi_thread();
  void inverse_matrix_GPU();

  Matrix get_result();

 private:
  Matrix a, output;
  unsigned int num_of_cores = 0;
};
