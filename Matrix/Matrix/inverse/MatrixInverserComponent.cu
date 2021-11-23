#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MatrixInverserComponent.h"
#include <stdexcept>
#include <thread>
#include <vector>
#include "../utils/matrix/MatrixUtils.h"
#include <iostream>

__global__ void gauss_jordan(float *matrix, float *identity, int dim, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dim && y < dim) {
    if (y != i) {
      identity[y * dim + x] -= identity[i * dim + x] * matrix[y * dim + i];
      if (x != i) {
        matrix[y * dim + x] -= matrix[i * dim + x] * matrix[y * dim + i];
      }
    }
  }
}

__global__ void no_diagonal_normalize(float *matrix, float *identity, int dim, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dim && y < dim)
    if (y == i && x != y) {
      identity[y * dim + x] /= matrix[i * dim + i];
      matrix[y * dim + x] /= matrix[i * dim + i];
    }
}

__global__ void diagonal_normalize(float *matrix, float *identity, int dim, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dim && y < dim)
    if (x == y && y == i) {
      identity[y * dim + x] /= matrix[i * dim + i];
      matrix[y * dim + x] /= matrix[i * dim + i];
    }
}

__global__ void set_zeros(float *matrix, float *identity, int dim, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dim && y < dim) {
    if (y != i) {
      if (x == i) {
        matrix[y * dim + x] = 0;
      }
    }
  }
}

__global__ void swap_rows(float *matrix, float *identity, int dim, int row1, int row2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dim && row1 < dim && row2 < dim && row1 == y) {
    float swap = matrix[row1 * dim + x];
    matrix[row1 * dim + x] = matrix[row2 * dim + x];
    matrix[row2 * dim + x] = swap;

    swap = identity[row1 * dim + x];
    identity[row1 * dim + x] = identity[row2 * dim + x];
    identity[row2 * dim + x] = swap;

  }
}

void MatrixInverserComponent::set_matrix(Matrix a) {
  if (a.get_x_dimension() != a.get_y_dimension()) {
    throw std::invalid_argument("matrix must be NxN.");
  }
  this->a = a;
  this->output = Matrix(a.get_matrix_name() + "_inverseOP",
                        a.get_x_dimension(), a.get_y_dimension());
  this->identity = MatrixUtils::create_identity(a.get_x_dimension());
}

void MatrixInverserComponent::inverse_matrix_CPU_single_thread() {
  size_t dim = a.get_x_dimension();

  for (int i = 0; i < dim; i++) {
    if (is_swapping && output[i][i] == 0) {
      bool swaped = false;
      for (int swap_id{i}; swap_id < dim; swap_id++) {
        if (output[swap_id][i] != 0) {
          cpu_swap_rows(i, swap_id);
          swaped = true;
          break;
        }
      }
      if (!swaped) break;
    }

    // output.display();
    // std::cout << "\n\n";
    cpu_row_normalize(i);
    cpu_gauss_jordan(i);
    // cpu_set_zeros(i);
  }
}

void MatrixInverserComponent::inverse_matrix_CPU_multi_thread() {
  size_t dim = a.get_x_dimension();
  num_of_cores = std::thread::hardware_concurrency();
  if (!num_of_cores) num_of_cores = 4;

  for (int i = 0; i < dim; i++) {
    if (is_swapping && output[i][i] == 0) {
      bool swaped = false;
      for (int swap_id{i}; swap_id < dim; swap_id++) {
        if (output[swap_id][i] != 0) {
          cpu_swap_rows_mt(i, swap_id);
          swaped = true;
          break;
        }
      }
      if (!swaped) break;
    }
    // output.display();
    // std::cout << "\n\n";

    cpu_row_normalize_mt(i);
    cpu_gauss_jordan_mt(i);
    // cpu_set_zeros_mt(i);
  }
}

void MatrixInverserComponent::inverse_matrix_GPU() {

  size_t dim = a.get_x_dimension();

  float *a_GPU_pointer, *out_GPU_pointer;
  int num_of_bytes = dim * dim * sizeof(float);

  cudaMalloc((void **)&a_GPU_pointer, num_of_bytes);
  cudaMalloc((void **)&out_GPU_pointer, num_of_bytes);

  cudaMemcpy(a_GPU_pointer, a[0], num_of_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(out_GPU_pointer, identity[0], num_of_bytes, cudaMemcpyHostToDevice);

  int sq = 16;
  dim3 dim_grid(dim / sq + 1, dim / sq + 1);
  dim3 dim_block(sq, sq);

  int dimm1 = dim - 1;
  for (int i = 0; i < dim; i++) {
    
    if (is_swapping && output[i][i] == 0) {
      bool swaped = false;
      for (int swap_id{i}; swap_id < dim; swap_id++) {
        if (output[swap_id][i] != 0) {
          swap_rows<<<dim_grid, dim_block>>>(a_GPU_pointer, out_GPU_pointer,
                                             dim, i, swap_id);
          swaped = true;
          break;
        }
      }
      if (!swaped) break;
    }


    no_diagonal_normalize<<<dim_grid, dim_block>>>(a_GPU_pointer,
                                                   out_GPU_pointer, dim, i);
    diagonal_normalize<<<dim_grid, dim_block>>>(a_GPU_pointer,
                                                   out_GPU_pointer,
                                                dim, i);
    gauss_jordan<<<dim_grid, dim_block>>>(a_GPU_pointer,
                                                        out_GPU_pointer, dim,
                                                        i);
    set_zeros<<<dim_grid, dim_block>>>(a_GPU_pointer, out_GPU_pointer, dim, i);
    if (is_swapping && i != dimm1 || true) {
      cudaMemcpy(output[0], a_GPU_pointer, num_of_bytes,
                 cudaMemcpyDeviceToHost);
    }
  }

  cudaMemcpy(identity[0], out_GPU_pointer, num_of_bytes, cudaMemcpyDeviceToHost);
  cudaFree(a_GPU_pointer);
  cudaFree(out_GPU_pointer);
}

void MatrixInverserComponent::prepare() {
  this->output = a;
  this->output.set_matrix_name(a.get_matrix_name() + "_inverseOP");
  this->identity = MatrixUtils::create_identity(a.get_x_dimension());
  this->identity.set_matrix_name(a.get_matrix_name() + "_inverseOP");
}

Matrix MatrixInverserComponent::get_result() { 
  Matrix o = identity;
  identity.set_matrix_name(a.get_matrix_name() + "_inverseOP");
  return o;
}

void MatrixInverserComponent::cpu_gauss_jordan(int iteration) {
  size_t dim = a.get_x_dimension();
  for (size_t y{0}; y < dim; y++) {
    float it_cell = output[y][iteration];
    for (size_t x{0}; x < dim; x++) {
      if (y != iteration) {
        identity[y][x] -= identity[iteration][x] * it_cell;
        if (x != iteration) {
          output[y][x] -= output[iteration][x] * it_cell;
        } else {
          output[y][x] = 0;
        }
      }
    }
  }
}

void MatrixInverserComponent::cpu_row_normalize(int iteration) {
  size_t dim = a.get_x_dimension();
  float cell_diag = output[iteration][iteration];
  for (size_t x{0}; x < dim; x++) {
    identity[iteration][x] /= cell_diag;
    output[iteration][x] /= cell_diag;
  }
}

void MatrixInverserComponent::cpu_set_zeros(int iteration) {
  size_t dim = a.get_x_dimension();
  for (size_t y{0}; y < dim; y++) {
    if (y != iteration) {
        output[y][iteration] = 0;
    }
  }
}

void MatrixInverserComponent::cpu_swap_rows(int row1, int row2) {
  size_t dim = a.get_x_dimension();
  for (size_t x{0}; x < dim; x++) {
    float swap = output[row1][x];
    output[row1][x] = output[row2][x];
    output[row2][x] = swap;
    
    swap = identity[row1][ x];
    identity[row1][x] = identity[row2][x];
    identity[row2][x] = swap;
  }
}

void MatrixInverserComponent::cpu_gauss_jordan_mt(int iteration) {
  std::vector<std::thread> threads;
  for (int i{0}; i < num_of_cores; i++) {
    threads.push_back(std::thread(
        &MatrixInverserComponent::cpu_gauss_jordan_thread, this, iteration, i));
  }

  for (int i{0}; i < num_of_cores; i++) {
    threads[i].join();
  }
}

void MatrixInverserComponent::cpu_row_normalize_mt(int iteration) {
  std::vector<std::thread> threads;
  for (int i{0}; i < num_of_cores; i++) {
    threads.push_back(
        std::thread(&MatrixInverserComponent::cpu_row_normalize_thread, this,
                    iteration, i, output[iteration][iteration]));
  }

  for (int i{0}; i < num_of_cores; i++) {
    threads[i].join();
  }
}

void MatrixInverserComponent::cpu_set_zeros_mt(int iteration) {
  std::vector<std::thread> threads;
  for (int i{0}; i < num_of_cores; i++) {
    threads.push_back(std::thread(
        &MatrixInverserComponent::cpu_set_zeros_thread, this, iteration, i));
  }

  for (int i{0}; i < num_of_cores; i++) {
    threads[i].join();
  }
}

void MatrixInverserComponent::cpu_swap_rows_mt(int row1, int row2) {
  std::vector<std::thread> threads;
  for (int i{0}; i < num_of_cores; i++) {
    threads.push_back(
        std::thread(&MatrixInverserComponent::cpu_swap_rows_thread, this, row1, row2, i));
  }

  for (int i{0}; i < num_of_cores; i++) {
    threads[i].join();
  }
}

void MatrixInverserComponent::cpu_gauss_jordan_thread(int iteration,
                                                      size_t thread_num) {
  size_t dim = a.get_x_dimension();
  for (size_t y{thread_num}; y < dim; y += num_of_cores) {
    float it_cell = output[y][iteration];
    for (size_t x{0}; x < dim; x++) {
      if (y != iteration) {
        identity[y][x] -= identity[iteration][x] * it_cell;
        if (x != iteration) {
          output[y][x] -= output[iteration][x] * it_cell;
        } else {
          output[y][x] = 0;
        }
      }
    }
  }
}

void MatrixInverserComponent::cpu_row_normalize_thread(int iteration,
                                                       size_t thread_num, float diag_cell) {
  size_t dim = a.get_x_dimension();
  size_t last = (thread_num + 1) *num_of_cores;
  for (size_t i{thread_num * num_of_cores}; i < dim && i < last; i++) {
    identity[iteration][i] /= diag_cell;
    output[iteration][i] /= diag_cell;
  }
}

void MatrixInverserComponent::cpu_set_zeros_thread(int iteration,
                                                   size_t thread_num) {
  size_t dim = a.get_x_dimension();
  size_t last = (thread_num + 1) * num_of_cores;
  for (size_t i{thread_num * num_of_cores}; i < dim && i < last; i++) {
    if (i != iteration) {
      output[i][iteration] = 0;
    }
  }
}

void MatrixInverserComponent::cpu_swap_rows_thread(int row1, int row2,
                                                   size_t thread_num) {
  size_t dim = a.get_x_dimension();
  size_t last = (thread_num + 1) * num_of_cores;
  for (size_t i{thread_num * num_of_cores}; i < dim && i < last; i++) {
    float swap = output[row1][i];
    output[row1][i] = output[row2][i];
    output[row2][i] = swap;

    swap = identity[row1][i];
    identity[row1][i] = identity[row2][i];
    identity[row2][i] = swap;
  }
}
