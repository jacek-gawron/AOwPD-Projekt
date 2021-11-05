#include "MatrixAdderComponent.h"
#include <stdexcept>
#include <thread>
#include <vector>

void MatrixAdderComponent::set_matrices(Matrix a, Matrix b) { 
  if (a.get_x_dimension() != b.get_x_dimension() ||
      a.get_y_dimension() != b.get_y_dimension()) {
    throw std::invalid_argument("Dimensions of amtrices must be the same.");
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

