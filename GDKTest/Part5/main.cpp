#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <sstream>

#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#include "nlohmann/json.hpp"
#include "Matrix.h"
#include "ModelBase.h"
#include "Model.h"

using json = nlohmann::json;

// ================== 原有代码 ==================
std::vector<int> getMatrixShape(const json& meta, const std::string& key) {
    return meta[key].get<std::vector<int>>();
}

std::string getDataType(const json& meta) {
    if (meta.contains("type")) {
        return meta["type"].get<std::string>();
    }
    return "fp32";
}

std::unique_ptr<ModelBase> createModel(const std::string& folderPath) {
    std::string jsonPath = folderPath + "/meta.json";

    std::ifstream meta_file(jsonPath);
    if (!meta_file.is_open()) {
        throw std::runtime_error("无法打开 meta.json");
    }

    json meta;
    meta_file >> meta;
    meta_file.close();

    auto w1_shape = getMatrixShape(meta, "fc1.weight");
    auto b1_shape = getMatrixShape(meta, "fc1.bias");
    auto w2_shape = getMatrixShape(meta, "fc2.weight");
    auto b2_shape = getMatrixShape(meta, "fc2.bias");

    bool useDouble = (getDataType(meta) == "fp64");

    if (useDouble) {
        Matrix<double> w1 = Matrix<double>::loadFromFile(folderPath + "/fc1.weight", w1_shape[0], w1_shape[1]);
        Matrix<double> b1 = Matrix<double>::loadFromFile(folderPath + "/fc1.bias", b1_shape[0], b1_shape[1]);
        Matrix<double> w2 = Matrix<double>::loadFromFile(folderPath + "/fc2.weight", w2_shape[0], w2_shape[1]);
        Matrix<double> b2 = Matrix<double>::loadFromFile(folderPath + "/fc2.bias", b2_shape[0], b2_shape[1]);

        return std::make_unique<Model<double>>(w1, b1, w2, b2);
    } else {
        Matrix<float> w1 = Matrix<float>::loadFromFile(folderPath + "/fc1.weight", w1_shape[0], w1_shape[1]);
        Matrix<float> b1 = Matrix<float>::loadFromFile(folderPath + "/fc1.bias", b1_shape[0], b1_shape[1]);
        Matrix<float> w2 = Matrix<float>::loadFromFile(folderPath + "/fc2.weight", w2_shape[0], w2_shape[1]);
        Matrix<float> b2 = Matrix<float>::loadFromFile(folderPath + "/fc2.bias", b2_shape[0], b2_shape[1]);

        return std::make_unique<Model<float>>(w1, b1, w2, b2);
    }
}

int main() {
    try {
        std::string folderPath = "mnist-fc";
        auto model = createModel(folderPath);

        // 初始化 Winsock
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);

        SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);

        sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_port = htons(8080);
        address.sin_addr.s_addr = INADDR_ANY;

        bind(server_fd, (sockaddr*)&address, sizeof(address));
        listen(server_fd, 3);

        std::cout << "Server started at port 8080..." << std::endl;

        while (true) {
            SOCKET client_socket = accept(server_fd, NULL, NULL);
            std::cout << "Client connected" << std::endl;

            char buffer[65536] = {0};
            int valread = recv(client_socket, buffer, sizeof(buffer), 0);

            std::stringstream ss(buffer);

            int rows, cols;
            ss >> rows >> cols;

            Matrix<float> input(rows, cols);
            for (int i = 0; i < rows * cols; i++) {
                ss >> input.data[i];
            }

            std::cout << "收到矩阵: " << rows << "x" << cols << std::endl;

            // 推理
            auto start = std::chrono::high_resolution_clock::now();
            Matrix<float> output = model->forward(input);
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "推理完成，用时: "
                      << std::chrono::duration<double>(end - start).count()
                      << " 秒\n";

            // 返回结果
            std::stringstream out;
            out << output.rows << " " << output.cols << "\n";
            for (int i = 0; i < output.rows * output.cols; i++) {
                out << output.data[i] << " ";
            }

            std::string result = out.str();
            send(client_socket, result.c_str(), result.size(), 0);

            closesocket(client_socket);
        }

        closesocket(server_fd);
        WSACleanup();

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
    }

    return 0;
}