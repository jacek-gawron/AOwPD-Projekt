#pragma once
#include "utils/IOManager/MatrixIOManager.hpp"
#include "adding/MatrixAdderComponent.h"
#include "multiplying/MatrixMultiplierComponent.h"
#include "transposing/MatrixTransposerComponent.h"
#include "inverse/MatrixInverserComponent.h"
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
    MatrixInverserComponent inverser;
    MatrixMultiplierComponent multiplier;
    MatrixTransposerComponent transposer;
    MatrixIOManager ioManager;

    std::vector<Matrix> loadedMatrices;
};