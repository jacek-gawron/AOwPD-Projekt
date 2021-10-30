
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "userInterface.hpp"

int main()
{
    UserInterface ui = UserInterface();
    printf("Detecting GPU...\n");
    if (ui.detectGpu()) {
        ui.printMainMenu();
    }
    else {
        printf("No CUDA device found\n");
    }
    return EXIT_SUCCESS;
}
