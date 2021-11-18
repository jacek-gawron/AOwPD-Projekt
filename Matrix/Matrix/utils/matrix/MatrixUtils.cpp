#include "MatrixUtils.h"

Matrix MatrixUtils::create_identity(size_t dim) { 
  Matrix m = create_zeros(dim, dim);
  m.set_matrix_name("identity");

  for (int i = 0; i < dim; i++) m[i][i] = 1.f;

  return m;
}

Matrix MatrixUtils::create_zeros(size_t dim_x, size_t dim_y) {
  Matrix m = Matrix("zeeros", dim_x, dim_y);
  for (int i = 0; i < dim_y; i++) {
    for (int j = 0; j < dim_x; j++) {
      m[i][j] = 0.f;
    }
  }

  return m;
}
