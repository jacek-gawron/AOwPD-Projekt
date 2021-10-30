#pragma once

class UserInterface {
public:
    UserInterface();
    ~UserInterface() = default;
    void printMainMenu();
    void printTestMenu();
    bool detectGpu();
};