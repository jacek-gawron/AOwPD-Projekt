#pragma once
#include <string>
#include <vector>

class Matrix {
 public:
  Matrix(std::string matrix_name = "Matrix", int dimension_x = 1,
         int dimension_y = 1);
  Matrix(std::string matrix_name, int dimension_x, int dimension_y,
         std::vector<std::vector<float>> arr);
  ~Matrix();

  Matrix(const Matrix& src);
  inline int get_x_dimension() const { return dim_x; }
  inline int get_y_dimension() const { return dim_y; }
  inline std::string get_matrix_name() const { return matrix_name; }
  inline void set_matrix_name(std::string str) { matrix_name = str; }

  float* operator[](int index);
  Matrix& operator=(const Matrix& src);

  void display();

 private:
  float** matrix{nullptr};
  int dim_x, dim_y;

  std::string matrix_name;

  void fill_from(const Matrix& src);

};
