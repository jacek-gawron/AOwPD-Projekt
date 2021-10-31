#include "MatrixIOManager.hpp"
#include <vector>
#include <sstream>
#include <iostream>

Matrix MatrixIOManager::loadMatrix(std::string path) {
  std::ifstream in{path};
  size_t ind = path.find_last_of('/');
  std::string matrix_name;
  if (ind == std::string::npos)
    matrix_name = path;
  else
    matrix_name = path.substr(ind);

  ind = matrix_name.find_last_of('.');
  if (ind != std::string::npos) matrix_name = matrix_name.substr(0, ind);

  
  int max_row_size = 0;
  if (in.good()) {
    std::vector<std::vector<float>> buff;
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream iss(line);
      std::vector<float> row;
      while (!iss.eof()) {
        float cell;
        iss >> cell;
        row.push_back(cell);
      }
      if (row.size() > max_row_size) max_row_size = row.size();
      buff.push_back(row);
    }
    for (auto& row : buff) {
      row.resize(max_row_size, .0f);
    }

    return Matrix(matrix_name, max_row_size, buff.size(), buff);
  }

  return Matrix();
 
}

void MatrixIOManager::saveMatrix(Matrix& matrix) {
  std::ofstream out = std::ofstream(matrix.get_matrix_name() + ".txt");
  for (int i = 0; i < matrix.get_y_dimension(); i++) {
    for (int j = 0; j < matrix.get_x_dimension(); j++) {
      out << matrix[i][j] << " ";
    }
    out << std::endl;
  }

}
