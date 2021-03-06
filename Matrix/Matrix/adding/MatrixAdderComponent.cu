#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MatrixAdderComponent.h"
#include <stdexcept>
#include <thread>
#include <vector>

__global__ void add_GPU(float *a, float *b, float *out, size_t dim_x,
                        size_t dim_y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  // int i = threadIdx.x;
  // int j = threadIdx.y;

  if (i < dim_x && j < dim_y) {
    out[j * dim_y + i] = a[j * dim_y + i] + b[j * dim_y + i];
  }
}

void MatrixAdderComponent::set_matrices(Matrix a, Matrix b) { 
  if (a.get_x_dimension() != b.get_x_dimension() ||
      a.get_y_dimension() != b.get_y_dimension()) {
    throw std::invalid_argument("Dimensions of matrices must be the same.");
  }
  this->a = a;
  this->b = b;
  this->output =
      Matrix(a.get_matrix_name() + "_" + b.get_matrix_name() + "_addOP",
                        a.get_x_dimension(), a.get_y_dimension());
}

void MatrixAdderComponent::add_matrices_CPU_single_thread() {
  int max_x = output.get_x_dimension(), max_y = output.get_y_dimension();
  for (int i{0}; i < max_y; i++) {
    for (int j{0}; j < max_x; j++) {
      output[i][j] = a[i][j] + b[i][j];
    }
  }
 }

void MatrixAdderComponent::add_matrices_CPU_multi_thread() {
  num_of_cores = std::thread::hardware_concurrency();
  if (!num_of_cores) num_of_cores = output.get_y_dimension();
   std::vector<std::thread> threads;

  for (int i{ 0 }; i < num_of_cores; i++) {
     threads.push_back(std::thread(&MatrixAdderComponent::thread_CPU_fun, this, i));
  }

  for (int i{0}; i < num_of_cores; i++) {
    threads[i].join();
  }
 }

void MatrixAdderComponent::add_matrices_GPU() {
  float *a_GPU_pointer, *b_GPU_pointer, *out_GPU_pointer;
  int num_of_bytes =
      output.get_x_dimension() * output.get_y_dimension() * sizeof(float);
  cudaMalloc((void **)&a_GPU_pointer, num_of_bytes);
  cudaMalloc((void **)&b_GPU_pointer, num_of_bytes);
  cudaMalloc((void **)&out_GPU_pointer, num_of_bytes);
  
  cudaMemcpy(a_GPU_pointer, a[0], num_of_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(b_GPU_pointer, b[0], num_of_bytes, cudaMemcpyHostToDevice);

  int sq = 16;
  dim3 dim_grid(output.get_x_dimension() / sq + 1,
                output.get_y_dimension() / sq + 1);
  dim3 dim_block(sq, sq);

  
  add_GPU<<<dim_grid, dim_block>>>(
      a_GPU_pointer, b_GPU_pointer, out_GPU_pointer,
                     output.get_x_dimension(), output.get_y_dimension());
  cudaMemcpy(output[0], out_GPU_pointer, num_of_bytes, cudaMemcpyDeviceToHost);

  cudaFree(a_GPU_pointer);
  cudaFree(b_GPU_pointer);
  cudaFree(out_GPU_pointer);
 }

Matrix MatrixAdderComponent::get_result() { return output; }

void MatrixAdderComponent::thread_CPU_fun(int thread_id) {
  int max_x = output.get_x_dimension(), max_y = output.get_y_dimension();
  for (int i{thread_id}; i < max_y; i += num_of_cores) {
    for (int j{0}; j < max_x; j++) {
      output[i][j] = a[i][j] + b[i][j];
    } 
  }
}
