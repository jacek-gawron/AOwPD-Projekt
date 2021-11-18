#pragma once
#include "Matrix.hpp"


class MatrixUtils {
 public:

  static Matrix create_identity(size_t dim);
  static Matrix create_zeros(size_t dim_x, size_t dim_y);
};
