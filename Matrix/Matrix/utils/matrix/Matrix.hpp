#pragma once
#include <string>
#include <vector>

class Matrix {
 public:
  Matrix(std::string matrix_name = "Matrix", size_t dimension_x = 1,
         size_t dimension_y = 1);
  Matrix(std::string matrix_name, size_t dimension_x, size_t dimension_y,
         std::vector<std::vector<float>> arr);
  ~Matrix();

  Matrix(const Matrix& src);
  inline size_t get_x_dimension() const { return dim_x; }
  inline size_t get_y_dimension() const { return dim_y; }
  inline std::string get_matrix_name() const { return matrix_name; }
  inline void set_matrix_name(std::string str) { matrix_name = str; }
  

  float* operator[](int index);
  Matrix& operator=(const Matrix& src);

  void display();

 private:
  float* matrix{nullptr};
  size_t dim_x, dim_y;

  std::string matrix_name;

  void fill_from(const Matrix& src);

};
