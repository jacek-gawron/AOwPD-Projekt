#include "MatrixInverseComponent.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>
#include <thread>
#include <vector>


void MatrixInverseComponent::set_matrix(Matrix a) {
  if (a.get_x_dimension() != a.get_y_dimension()) {
    throw std::invalid_argument("matrix must be NxN.");
  }
  this->a = a;
  this->output = Matrix(a.get_matrix_name() + "inverseOP",
                        a.get_x_dimension(), a.get_y_dimension());
}

void MatrixInverseComponent::inverse_matrix_CPU_single_thread() {}

void MatrixInverseComponent::inverse_matrix_CPU_multi_thread() {}

void MatrixInverseComponent::inverse_matrix_GPU() {}

Matrix MatrixInverseComponent::get_result() { return output; }
