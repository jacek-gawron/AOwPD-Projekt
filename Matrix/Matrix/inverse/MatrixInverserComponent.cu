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
      if (/* x != i*/ true) {
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
        // matrix[y * dim + x] = 0;
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
  this->output = Matrix(a.get_matrix_name() + "inverseOP",
                        a.get_x_dimension(), a.get_y_dimension());
  this->identity = MatrixUtils::create_identity(a.get_x_dimension());
}

void MatrixInverserComponent::inverse_matrix_CPU_single_thread() {}

void MatrixInverserComponent::inverse_matrix_CPU_multi_thread() {}

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
    /*
    if (output[i][i] == 0) {
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
    }*/

    no_diagonal_normalize<<<dim_grid, dim_block>>>(a_GPU_pointer,
                                                   out_GPU_pointer, dim, i);
    diagonal_normalize<<<dim_grid, dim_block>>>(a_GPU_pointer,
                                                   out_GPU_pointer,
                                                dim, i);
    gauss_jordan<<<dim_grid, dim_block>>>(a_GPU_pointer,
                                                        out_GPU_pointer, dim,
                                                        i);
    set_zeros<<<dim_grid, dim_block>>>(a_GPU_pointer, out_GPU_pointer, dim, i);
    /* if (i != dimm1) {
      cudaMemcpy(output[0], a_GPU_pointer, num_of_bytes,
                 cudaMemcpyDeviceToHost);
    }*/
  }

  cudaMemcpy(output[0], out_GPU_pointer, num_of_bytes, cudaMemcpyDeviceToHost);
  cudaFree(a_GPU_pointer);
  cudaFree(out_GPU_pointer);
}

Matrix MatrixInverserComponent::get_result() { return output; }
