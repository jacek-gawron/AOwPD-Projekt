#pragma once
#include "../utils/matrix/Matrix.hpp"

class MatrixInverserComponent {
 public:
  void set_matrix(Matrix a);

  unsigned int get_num_of_threads() { return num_of_cores; }

  void inverse_matrix_CPU_single_thread();
  void inverse_matrix_CPU_multi_thread();
  void inverse_matrix_GPU();
  inline void enable_swapping() { is_swapping = true; }
  inline void disable_swapping() { is_swapping = false; }
  void prepare();

  Matrix get_result();

 private:
  bool is_swapping = false;
  Matrix a, output, identity;
  unsigned int num_of_cores = 0;

  void cpu_gauss_jordan(int iteration);
  void cpu_row_normalize(int iteration);
  void cpu_set_zeros(int iteration);
  void cpu_swap_rows(int row1, int row2);

  void cpu_gauss_jordan_mt(int iteration);
  void cpu_row_normalize_mt(int iteration);
  void cpu_set_zeros_mt(int iteration);
  void cpu_swap_rows_mt(int row1, int row2);

  void cpu_gauss_jordan_thread(int iteration, size_t thread_num);
  void cpu_row_normalize_thread(int iteration, size_t thread_num, float diag_cell);
  void cpu_set_zeros_thread(int iteration, size_t thread_num);
  void cpu_swap_rows_thread(int row1, int row2, size_t thread_num);
};
