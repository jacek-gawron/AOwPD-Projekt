#include "./Matrix.hpp"
#include <iostream>


Matrix::Matrix(std::string matrix_name, int dimension_x, int dimension_y): 
  matrix_name {matrix_name},
  dim_x{dimension_x}, 
  dim_y{dimension_y} {

  this->matrix = new float*[dim_y];
  for (int i = 0; i < dim_y; i++) {
    this->matrix[i] = new float[dim_x];
  }
}

Matrix::Matrix(std::string matrix_name, int dimension_x, int dimension_y,
               std::vector<std::vector<float>> arr)
    : Matrix(matrix_name, dimension_x, dimension_y) {
  for (int i = 0; i < dim_y; i++) {
    bool fill_zeros_row = (i >= arr.size());
    for (int j = 0; j < dim_x; j++) {
      bool fill_zeros_cell = (j >= arr[i].size());
      if (fill_zeros_row || fill_zeros_cell) {
        matrix[i][j] = 0.0f;
      } else {
        matrix[i][j] = arr[i][j];
      }
    }
  }

}

Matrix::~Matrix(){
  for (int i = 0; i < dim_y; i++) {
    delete[] this->matrix[i];
  }
  delete[] this->matrix;
}

Matrix::Matrix(const Matrix& src): 
  Matrix(src.matrix_name, src.dim_x, src.dim_y) {
  this->fill_from(src);
  
}

float* Matrix::operator[](int index) { 
  return this->matrix[index];
}

Matrix& Matrix::operator=(const Matrix& src) {
  
  if (&src != this) {
    this->~Matrix();
    this->dim_x = src.dim_x;
    this->dim_y = src.dim_y;
    this->matrix_name = src.matrix_name;

    this->matrix = new float*[dim_y];
    for (int i = 0; i < dim_y; i++) {
      this->matrix[i] = new float[dim_x];
    }
    this->fill_from(src);
  }
  return *this;
}

void Matrix::display() {
  for (int i = 0; i < dim_y; i++) {
    for (int j = 0; j < dim_x; j++) {
      std::cout << matrix[i][j] << '\t';
    }
    std::cout << std::endl;
  }

}

void Matrix::fill_from(const Matrix& src) {
  for (int i = 0; i < this->dim_y; ++i) {
    for (int j = 0; j < this->dim_x; ++j) {
      this->matrix[i][j] = src.matrix[i][j];
    }
  }
}
