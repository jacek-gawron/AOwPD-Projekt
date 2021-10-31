#pragma once
#include <string>
#include <fstream>
#include "../matrix/Matrix.hpp"


class MatrixIOManager {
 public:
  Matrix loadMatrix(std::string path);
  void saveMatrix(Matrix& matrix);
};
