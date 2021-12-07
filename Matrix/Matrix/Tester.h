#pragma once
#include "utils/IOManager/MatrixIOManager.hpp"
#include "adding/MatrixAdderComponent.h"
#include "multiplying/MatrixMultiplierComponent.h"
#include "transposing/MatrixTransposerComponent.h"
#include "inverse/MatrixInverserComponent.h"
#include <vector>
#include <iostream>

class Tester {
 public:
  void testFromConfig(std::string filePath);

 private:
  MatrixAdderComponent adder;
  MatrixInverserComponent inverser;
  MatrixMultiplierComponent multiplier;
  MatrixTransposerComponent transposer;
  MatrixIOManager ioManager;

  void addition(size_t count, size_t op1, size_t op2,
                std::string output_file_name);
  void multiplication(size_t count, size_t op1, size_t op2,
                      std::string output_file_name);
  void inverse(size_t count, size_t op1,
               std::string output_file_name);
  void transpose(size_t count, size_t op1,
                 std::string output_file_name);

  std::vector<Matrix> loadedMatrices;
};
