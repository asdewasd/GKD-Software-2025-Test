#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "nlohmann/json.hpp"
#include "Matrix.h"
#include "ModelBase.h"
#include "Model.h"

using json = nlohmann::json;

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

        Matrix<float> input(1, 784);
        for (int i = 0; i < 784; ++i) {
            input(0, i) = 0.5f;
        }

        std::cout << "=== Forward 开始 ===" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        Matrix<float> output = model->forward(input);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Forward 用时: " << elapsed.count() << " 秒\n";

        std::cout << "输出维度: " << output.rows << " x " << output.cols << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
    }

    return 0;
}