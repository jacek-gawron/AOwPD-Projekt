#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <windows.h>
#include <iostream>
#include <driver_types.h>
#include <cuda_runtime.h>
#include "userInterface.hpp"
#include "cudaStatus.hpp"
#include "Tester.h"

UserInterface::UserInterface() {

}

bool UserInterface::detectGpu() {
    int devicesCount = 0;
    CUDA_STATUS(cudaGetDeviceCount(&devicesCount));
    return devicesCount > 0;
   // return true;
}

void UserInterface::printTestMenu() {
  return;
    std::string menu = "---Operations on Matrices TESTS---\n";
    menu.append("[1]. Test addition\n");
    menu.append("[2]. Test multiplication\n");
    menu.append("[3]. Test transposition\n");
    menu.append("[4]. Test inverse\n");
    menu.append("[0]. Exit test menu\n");
    menu.append(">");
    int cnt = 0;

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
        cnt++;
    } while (result != '0' && cnt < 100000);
}

void UserInterface::printMainMenu() {
  int cnt = 0;
    std::string menu = "---Operations on Matrices---\n";
    menu.append("[1]. Load matrices\n");
    menu.append("[2]. Print matrices\n");
    menu.append("[3]. Matrix addition\n");
    menu.append("[4]. Matrix multiplication\n");
    menu.append("[5]. Matrix transposition\n");
    menu.append("[6]. Matrix inverse\n");
    menu.append("[7]. Test menu\n");
    menu.append("[8]. Test from file\n");
    menu.append("[0]. Exit\n");
    menu.append(">");


    char result = 'a';
    do {
        rewind(stdin);
        printf("%s", menu.c_str());
        //scanf("%c", &result);
        std::cin >> result;
        switch (result) {
        case '1': {
            std::cout << "Podaj plik z macierza\n>";
            try {
              std::string filePath;
              std::cin >> filePath;
              Matrix m = ioManager.loadMatrix(filePath);
              loadedMatrices.push_back(m);
            } catch (...) {
              std::cout << "Wst?pi? blad" << std::endl;
            }
           
            // m.display();
            break;
        }
        case '2': {
            printf("print\n");

            for (int i{0}; i < loadedMatrices.size(); i++) {
                printf("Matrix: %d \tName: %s\n", i, loadedMatrices[i].get_matrix_name().c_str());
                loadedMatrices[i].display();
                printf("\n\n");
            }
            break;
        }
        case '3': {
            int aId, bId;
          Matrix m;
            printf("add\n");
            printf("Index of first matrix\n>");
            scanf("%d", &aId);
            printf("Index of second matrix\n>");
            scanf("%d", &bId);
            try {
                printf("Matrix %d:\n", aId);
                loadedMatrices[aId].display();
                printf("\nMatrix %d:\n", bId);
                loadedMatrices[bId].display();
                printf("\n\n");

                adder.set_matrices(loadedMatrices[aId], loadedMatrices[bId]);
               
                adder.add_matrices_CPU_single_thread();
                m = adder.get_result();
                m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU");
                ioManager.saveMatrix(m);
                printf("Single thread result\n");
                m.display();
                printf("\n\n");

                adder.add_matrices_CPU_multi_thread();
                m = adder.get_result();
                m.set_matrix_name(m.get_matrix_name() +
                                  "_multiThreadCPU(threads-" +
                                  std::to_string(adder.get_num_of_threads()) + ")");
                ioManager.saveMatrix(m);
                printf("Multi thread result (threads: %d)\n",
                       adder.get_num_of_threads());
                m.display();
                printf("\n\n");
                
                adder.add_matrices_GPU();
                m = adder.get_result();
                m.set_matrix_name(
                    m.get_matrix_name() + "_GPU");
                ioManager.saveMatrix(m);
                printf("GPU result:\n");
                m.display();
                printf("\n\n");

            } catch (std::exception e) {
                printf("Error occured: %s", e.what());
            }
            break;
        }
        case '4': {
            int aId, bId;
            printf("multiply\n");
            printf("Index of first matrix\n>");
            scanf("%d", &aId);
            printf("Index of second matrix\n>");
            scanf("%d", &bId);
            try {
                printf("Matrix %d:\n", aId);
                loadedMatrices[aId].display();
                printf("\nMatrix %d:\n", bId);
                loadedMatrices[bId].display();
                printf("\n\n");

                multiplier.set_matrices(loadedMatrices[aId], loadedMatrices[bId]);

                multiplier.multiply_matrices_CPU_single_thread();
                Matrix m = multiplier.get_result();
                m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU");
                ioManager.saveMatrix(m);
                printf("Single thread result\n");
                m.display();
                printf("\n\n");


                multiplier.multiply_matrices_CPU_multi_thread();
                m = multiplier.get_result();
                m.set_matrix_name(m.get_matrix_name() +
                    "_multiThreadCPU(threads-" +
                    std::to_string(multiplier.get_num_of_threads()) + ")");
                ioManager.saveMatrix(m);
                printf("Multi thread result (threads: %d)\n",
                    multiplier.get_num_of_threads());
                m.display();
                printf("\n\n");

                multiplier.multiply_matrices_GPU();
                m = multiplier.get_result();
                m.set_matrix_name(m.get_matrix_name() + "_GPU");
                ioManager.saveMatrix(m);
                printf("GPU result\n",
                    multiplier.get_num_of_threads());
                m.display();
                printf("\n\n");

            }
            catch (std::exception e) {
                printf("Error occured: %s", e.what());
            }
            break;
        }
        case '5': {
            int aId;
            printf("transpose\n");
            printf("Index of matrix to transpose\n>");
            scanf("%d", &aId);
            try {
                printf("Matrix %d:\n", aId);
                loadedMatrices[aId].display();

                transposer.set_matrix(loadedMatrices[aId]);
                transposer.transpose_matrix_CPU_single_thread();
                Matrix m = transposer.get_result();
                m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU");
                ioManager.saveMatrix(m);
                printf("Single thread result\n");
                m.display();
                printf("\n\n");

                transposer.transpose_matrix_CPU_multi_thread();
                m = transposer.get_result();
                m.set_matrix_name(m.get_matrix_name() +
                    "_multiThreadCPU(threads-" +
                    std::to_string(transposer.get_num_of_threads()) + ")");
                ioManager.saveMatrix(m);
                printf("Multi thread result (threads: %d)\n",
                    transposer.get_num_of_threads());
                m.display();
                printf("\n\n");

                transposer.transpose_matrix_GPU();
                m = transposer.get_result();
                m.set_matrix_name(
                    m.get_matrix_name() + "_GPU");
                ioManager.saveMatrix(m);
                printf("GPU result:\n");
                m.display();
                printf("\n\n");
            }
            catch (std::exception err) {
                printf("Error occured: %s", err.what());
            }
            break;
        }
        case '6': {
            printf("inverse\n");

            int mId;
            printf("Index of matrix\n>");
            scanf("%d", &mId);
            Matrix m;
            try {
              printf("Matrix %d:\n", mId);
              //loadedMatrices[mId].display();
              printf("\n\n");

              inverser.set_matrix(loadedMatrices[mId]);
              /*
              inverser.add_matrices_CPU_single_thread();
              Matrix m = inverser.get_result();
              m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU");
              ioManager.saveMatrix(m);
              printf("Single thread result\n");
              m.display();
              printf("\n\n");
              */
              /*
              inverser.add_matrices_CPU_multi_thread();
              m = inverser.get_result();
              m.set_matrix_name(
                  m.get_matrix_name() + "_multiThreadCPU(threads: " +
                  std::to_string(inverser.get_num_of_threads()) + ")");
              ioManager.saveMatrix(m);
              printf("Multi thread result (threads: %d)\n",
                     inverser.get_num_of_threads());
              m.display();
              printf("\n\n");
              */
              inverser.prepare();
              inverser.enable_swapping();
              inverser.inverse_matrix_CPU_single_thread();
              m = inverser.get_result();
              m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU(swapping)");
              ioManager.saveMatrix(m);
              printf("Single thread result(swapping):\n");
              m.display();
              printf("\n\n");

              inverser.prepare();
              inverser.disable_swapping();
              inverser.inverse_matrix_CPU_single_thread();
              m = inverser.get_result();
              m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU(no_swapping)");
              ioManager.saveMatrix(m);
              printf("Single thread result(no_swapping):\n");
              m.display();
              printf("\n\n");



              inverser.prepare();
              inverser.enable_swapping();
              inverser.inverse_matrix_CPU_multi_thread();
              m = inverser.get_result();
              m.set_matrix_name(m.get_matrix_name() + "_multiThreadCPU(swapping; threads-" +
                  std::to_string(inverser.get_num_of_threads()) + ")");
              ioManager.saveMatrix(m);
              printf("Multi thread result(swapping; threads: %d):\n", inverser.get_num_of_threads());
              m.display();
              printf("\n\n");

              inverser.prepare();
              inverser.disable_swapping();
              inverser.inverse_matrix_CPU_multi_thread();
              m = inverser.get_result();
              m.set_matrix_name(m.get_matrix_name() +
                                "_multiThreadCPU(no_swapping; threads-" +
                                std::to_string(inverser.get_num_of_threads()) +
                                ")");
              ioManager.saveMatrix(m);
              printf("Multi thread result(no_swapping; threads: %d):\n",
                     inverser.get_num_of_threads());
              m.display();
              printf("\n\n");



              inverser.prepare();
              inverser.enable_swapping();
              inverser.inverse_matrix_GPU();
              m = inverser.get_result();
              m.set_matrix_name(m.get_matrix_name() + "_GPU(swapping)");
              ioManager.saveMatrix(m);
              printf("GPU result(swapping):\n");
              m.display();
              printf("\n\n");

              inverser.prepare();
              inverser.disable_swapping();
              inverser.inverse_matrix_GPU();
              m = inverser.get_result();
              m.set_matrix_name(m.get_matrix_name() + "_GPU(no_swapping)");
              ioManager.saveMatrix(m);
              printf("GPU result(no_swapping):\n");
              m.display();
              printf("\n\n");

            } catch (std::exception e) {
              printf("Error occured: %s", e.what());
            }
            break;
        }
        case '7': {
            system("cls");
            // printTestMenu();
            break;
        }
        case '8': {
            std::string filePath;
            std::cout << "Podaj sciezke do pliku z konfiguracja\n>";
            std::cin >> filePath;
            try {
                Tester().testFromConfig(filePath);
            } catch (...) {
                std::cout << "Wst?pi? blad" << std::endl;
            }
            
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
        cnt++;
    } while (result != '0' && cnt < 10000);
}