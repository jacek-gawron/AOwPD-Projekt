#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <windows.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include "userInterface.hpp"
#include "cudaStatus.hpp"

UserInterface::UserInterface() {

}

bool UserInterface::detectGpu() {
    int devicesCount = 0;
    CUDA_STATUS(cudaGetDeviceCount(&devicesCount));
    return devicesCount > 0;
}

void UserInterface::printTestMenu() {
    std::string menu = "---Operations on Matrices TESTS---\n";
    menu.append("[1]. Test addition\n");
    menu.append("[2]. Test multiplication\n");
    menu.append("[3]. Test transposition\n");
    menu.append("[4]. Test inverse\n");
    menu.append("[0]. Exit test menu\n");
    menu.append(">");


    char result = 'a';
    do {
        rewind(stdin);
        printf("%s", menu.c_str());
        scanf("%c", &result);
        switch (result) {
        case '1': {
            printf("test add\n");
            break;
        }
        case '2': {
            printf("test multiply\n");
            break;
        }
        case '3': {
            printf("test transposition\n");
            break;
        }
        case '4': {
            printf("test inverse\n");
            break;
        }
        case '0': {
            printf("Exiting test menu\n");
            rewind(stdin);
            break;
        }
        default: {
            printf("Invalid value\n");
        }
        }
        if (result != '0') {
            system("pause");
            system("cls");
        }
    } while (result != '0');
}

void UserInterface::printMainMenu() {
    std::string menu = "---Operations on Matrices---\n";
    menu.append("[1]. Load matrices\n");
    menu.append("[2]. Print matrices\n");
    menu.append("[3]. Matrix addition/subtraction\n");
    menu.append("[4]. Matrix multiplication\n");
    menu.append("[5]. Matrix transposition\n");
    menu.append("[6]. Matrix inverse\n");
    menu.append("[7]. Test menu\n");
    menu.append("[0]. Exit\n");
    menu.append(">");


    char result = 'a';
    do {
        rewind(stdin);
        printf("%s", menu.c_str());
        scanf("%c", &result);
        switch (result) {
        case '1': {
            printf("load\n");
            break;
        }
        case '2': {
            printf("print\n");
            break;
        }
        case '3': {
            printf("add\n");
            break;
        }
        case '4': {
            printf("multiply\n");
            break;
        }
        case '5': {
            printf("transpose\n");
            break;
        }
        case '6': {
            printf("inverse\n");
            break;
        }
        case '7': {
            system("cls");
            printTestMenu();
            break;
        }
        case '0': {
            printf("Exiting\n");
            break;
        }
        default: {
            printf("Invalid value\n");
        }
        }
        system("pause");
        system("cls");
    } while (result != '0');
}