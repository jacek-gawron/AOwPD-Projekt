#include "Tester.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <thread>

void Tester::testFromConfig(std::string filePath) {
  std::ifstream in{filePath};

  if (in.good()) {
    std::string buffer;
    do {
      in >> buffer;
      auto ind = buffer.find_first_of("###");
      if (ind != std::string::npos) break;

      Matrix m = ioManager.loadMatrix(buffer);
      loadedMatrices.push_back(m);

    } while (!in.eof());


    std::vector<std::vector<float>> buff;
    std::string ops, output_file_name;
    size_t count, op1, op2;
    
    while (!in.eof()) {
      in >> ops >> count >> op1 >> op2 >> output_file_name;

      if (!ops.compare("addition")) {
        adder.set_matrices(loadedMatrices[op1], loadedMatrices[op2]);
        addition(count, op1, op2, output_file_name);
      } else if (!ops.compare("multiplication")) {
        multiplier.set_matrices(loadedMatrices[op1], loadedMatrices[op2]);
        multiplication(count, op1, op2, output_file_name);
      } else if (!ops.compare("transposition")) {
        transposer.set_matrix(loadedMatrices[op1]);
        transpose(count, op1, output_file_name);
      } else if (!ops.compare("inverse")) {
        inverser.set_matrix(loadedMatrices[op1]);
        inverse(count, op1, output_file_name);
      }
    }
  }
}

void Tester::addition(size_t count, size_t op1, size_t op2,
                      std::string output_file_name) {

  std::ofstream out(output_file_name);
  std::string cols =
      std::string() + "CPU (single thread) [ms];CPU (multi thread)(" +
      std::to_string(std::thread::hardware_concurrency()) + ") [ms]; GPU [ms]";
  out << cols << std::endl;

  for (size_t i{0}; i < count; i++) {
    auto start = std::chrono::steady_clock::now();
    adder.add_matrices_CPU_single_thread();
    auto end = std::chrono::steady_clock::now();
    long long st =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = adder.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU_test");
      ioManager.saveMatrix(m);
    }


    start = std::chrono::steady_clock::now();
    adder.add_matrices_CPU_multi_thread();
    end = std::chrono::steady_clock::now();
    long long mt =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = adder.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_multiThreadCPU(threads-" +
                        std::to_string(adder.get_num_of_threads()) + ")_test");
      ioManager.saveMatrix(m);
    }


    start = std::chrono::steady_clock::now();
    adder.add_matrices_GPU();
    end = std::chrono::steady_clock::now();
    long long gput =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = adder.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_GPU_test");
      ioManager.saveMatrix(m);
    }


    out << st << ";" << mt << ";" << gput << std::endl;
  }

}

void Tester::multiplication(size_t count, size_t op1, size_t op2,
                            std::string output_file_name) {

  std::ofstream out(output_file_name);
  std::string cols =
      std::string() + "CPU (single thread) [ms];CPU (multi thread)(" +
      std::to_string(std::thread::hardware_concurrency()) + ") [ms]; GPU [ms]";
  out << cols << std::endl;

  for (size_t i{0}; i < count; i++) {
    auto start = std::chrono::steady_clock::now();
    multiplier.multiply_matrices_CPU_single_thread();
    auto end = std::chrono::steady_clock::now();
    long long st =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = multiplier.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU_test");
      ioManager.saveMatrix(m);
    }


    start = std::chrono::steady_clock::now();
    multiplier.multiply_matrices_CPU_multi_thread();
    end = std::chrono::steady_clock::now();
    long long mt =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = multiplier.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_multiThreadCPU(threads-" +
                        std::to_string(multiplier.get_num_of_threads()) + ")_test");
      ioManager.saveMatrix(m);
    }


    start = std::chrono::steady_clock::now();
    multiplier.multiply_matrices_GPU();
    end = std::chrono::steady_clock::now();
    long long gput =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = multiplier.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_GPU_test");
      ioManager.saveMatrix(m);
    }


    out << st << ";" << mt << ";" << gput << std::endl;
  }

}

void Tester::inverse(size_t count, size_t op1, std::string output_file_name) {
  std::ofstream out(output_file_name);
  std::string cols =
      std::string() +
      "CPU (single thread)(no swapping) [ms];CPU (multi thread)(" +
      std::to_string(std::thread::hardware_concurrency()) +
      ")(no swapping) [ms];" +
      "GPU(no swapping) [ms];CPU (single thread)(swapping) [ms];CPU (multi "
      "thread)(" +
      std::to_string(std::thread::hardware_concurrency()) +
      ")(swapping) [ms];" + "GPU(swapping) [ms]";
  out << cols << std::endl;

  for (size_t i{0}; i < count; i++) {
    inverser.prepare();
    inverser.disable_swapping();
    auto start = std::chrono::steady_clock::now();
    inverser.inverse_matrix_CPU_single_thread();
    auto end = std::chrono::steady_clock::now();
    long long st_ns =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = inverser.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU(no_swapping)_test");
      ioManager.saveMatrix(m);
    }


    inverser.prepare();
    inverser.disable_swapping();
    start = std::chrono::steady_clock::now();
    inverser.inverse_matrix_CPU_multi_thread();
    end = std::chrono::steady_clock::now();
    long long mt_ns =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = inverser.get_result();
      m.set_matrix_name(m.get_matrix_name() +
                        "_multiThreadCPU(no_swapping; threads-" +
                        std::to_string(inverser.get_num_of_threads()) + ")_test");
      ioManager.saveMatrix(m);
    }

    inverser.prepare();
    inverser.disable_swapping();
    start = std::chrono::steady_clock::now();
    inverser.inverse_matrix_GPU();
    end = std::chrono::steady_clock::now();
    long long gput_ns =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = inverser.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_GPU(no_swapping)_test");
      ioManager.saveMatrix(m);
    }


    ///////////////


    inverser.prepare();
    inverser.enable_swapping();
    start = std::chrono::steady_clock::now();
    inverser.inverse_matrix_CPU_single_thread();
    end = std::chrono::steady_clock::now();
    long long st_s =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = inverser.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU(swapping)_test");
      ioManager.saveMatrix(m);
    }


    inverser.prepare();
    inverser.enable_swapping();
    start = std::chrono::steady_clock::now();
    inverser.inverse_matrix_CPU_multi_thread();
    end = std::chrono::steady_clock::now();
    long long mt_s =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = inverser.get_result();
      m.set_matrix_name(m.get_matrix_name() +
                        "_multiThreadCPU(swapping; threads-" +
                        std::to_string(inverser.get_num_of_threads()) + ")_test");
      ioManager.saveMatrix(m);
    }


    inverser.prepare();
    inverser.enable_swapping();
    start = std::chrono::steady_clock::now();
    inverser.inverse_matrix_GPU();
    end = std::chrono::steady_clock::now();
    long long gput_s =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = inverser.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_GPU(swapping)_test");
      ioManager.saveMatrix(m);
    }


    out << st_ns << ";" << mt_ns << ";" << gput_ns << st_s << ";" << mt_s
        << ";" << gput_s  << std::endl;
  }
}

void Tester::transpose(size_t count, size_t op1, std::string output_file_name) {
  std::ofstream out(output_file_name);
  std::string cols =
      std::string() + "CPU (single thread) [ms];CPU (multi thread)(" +
      std::to_string(std::thread::hardware_concurrency()) + ") [ms]; GPU [ms]";
  out << cols << std::endl;

  for (size_t i{0}; i < count; i++) {
    auto start = std::chrono::steady_clock::now();
    transposer.transpose_matrix_CPU_single_thread();
    auto end = std::chrono::steady_clock::now();
    long long st =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = transposer.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_singleThreadCPU_test");
      ioManager.saveMatrix(m);
    }


    start = std::chrono::steady_clock::now();
    transposer.transpose_matrix_CPU_multi_thread();
    end = std::chrono::steady_clock::now();
    long long mt =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = transposer.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_multiThreadCPU(threads-" +
                        std::to_string(transposer.get_num_of_threads()) + ")_test");
      ioManager.saveMatrix(m);
    }


    start = std::chrono::steady_clock::now();
    transposer.transpose_matrix_GPU();
    end = std::chrono::steady_clock::now();
    long long gput =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    if (i == count - 1) {
      Matrix m = transposer.get_result();
      m.set_matrix_name(m.get_matrix_name() + "_GPU_test");
      ioManager.saveMatrix(m);
    }


    out << st << ";" << mt << ";" << gput << std::endl;
  }
}
