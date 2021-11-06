#pragma once
#include "utils/IOManager/MatrixIOManager.hpp"
#include "adding/MatrixAdderComponent.h"
#include "transposing/MatrixTransposerComponent.h"
#include <vector>

class UserInterface {
public:
    UserInterface();
    ~UserInterface() = default;
    void printMainMenu();
    void printTestMenu();
    bool detectGpu();

private:
    MatrixAdderComponent adder;
    MatrixTransposerComponent transposer;
    MatrixIOManager ioManager;

    std::vector<Matrix> loadedMatrices;
};