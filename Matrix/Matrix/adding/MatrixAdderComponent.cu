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
  this->output = Matrix(a.get_matrix_name() + b.get_matrix_name() + "addOP",
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
  const int num_of_thread_blocks =
      (output.get_x_dimension() * output.get_y_dimension() + 1) / 256;
  
  add_GPU<<<num_of_thread_blocks, 256>>>(
      a_GPU_pointer, b_GPU_pointer, out_GPU_pointer,
                     output.get_x_dimension(), output.get_y_dimension());
  cudaMemcpy(out_GPU_pointer, output[0], num_of_bytes, cudaMemcpyHostToDevice);
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
