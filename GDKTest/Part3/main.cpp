#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>
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
    return "fp32";  // 默认 float
}

std::unique_ptr<ModelBase> createModel(const std::string& folderPath) {
    std::string jsonPath = folderPath + "/meta.json";

    std::ifstream meta_file(jsonPath);
    if (!meta_file.is_open()) {
        throw std::runtime_error("无法打开 meta.json: " + jsonPath);
    }
    
    json meta;
    meta_file >> meta;
    meta_file.close();
    
    std::string dataType = getDataType(meta);
    std::cout << "=== 模型信息 ===" << std::endl;
    std::cout << "文件夹：" << folderPath << std::endl;
    std::cout << "数据类型：" << dataType << std::endl;

    auto w1_shape = getMatrixShape(meta, "fc1.weight");
    auto b1_shape = getMatrixShape(meta, "fc1.bias");
    auto w2_shape = getMatrixShape(meta, "fc2.weight");
    auto b2_shape = getMatrixShape(meta, "fc2.bias");

    bool useDouble = (dataType == "fp64" || dataType == "double" || 
                      folderPath.find("plus") != std::string::npos);
    
    if (useDouble) {
        std::cout << "加载 double 类型模型..." << std::endl;
        
        Matrix<double> w1 = Matrix<double>::loadFromFile(
            folderPath + "/fc1.weight", w1_shape[0], w1_shape[1]);
        Matrix<double> b1 = Matrix<double>::loadFromFile(
            folderPath + "/fc1.bias", b1_shape[0], b1_shape[1]);
        Matrix<double> w2 = Matrix<double>::loadFromFile(
            folderPath + "/fc2.weight", w2_shape[0], w2_shape[1]);
        Matrix<double> b2 = Matrix<double>::loadFromFile(
            folderPath + "/fc2.bias", b2_shape[0], b2_shape[1]);
        
        return std::make_unique<Model<double>>(w1, b1, w2, b2);
    } else {
        std::cout << "加载 float 类型模型..." << std::endl;
        
        Matrix<float> w1 = Matrix<float>::loadFromFile(
            folderPath + "/fc1.weight", w1_shape[0], w1_shape[1]);
        Matrix<float> b1 = Matrix<float>::loadFromFile(
            folderPath + "/fc1.bias", b1_shape[0], b1_shape[1]);
        Matrix<float> w2 = Matrix<float>::loadFromFile(
            folderPath + "/fc2.weight", w2_shape[0], w2_shape[1]);
        Matrix<float> b2 = Matrix<float>::loadFromFile(
            folderPath + "/fc2.bias", b2_shape[0], b2_shape[1]);
        
        return std::make_unique<Model<float>>(w1, b1, w2, b2);
    }
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== Part3 模板与多态测试 ===" << std::endl;
        std::cout << std::endl;

        std::string folderPath = "mnist-fc";
        if (argc > 1 && std::string(argv[1]) == "plus") {
            folderPath = "mnist-fc-plus";
        }
        std::cout << "使用文件夹：" << folderPath << std::endl;
        std::cout << std::endl;

        std::unique_ptr<ModelBase> model = createModel(folderPath);
        
        std::cout << "\n=== 模型信息 ===" << std::endl;
        std::cout << "类型：" << model->getModelType() << std::endl;
        model->printShape();

        Matrix<float> input(1, 784);
        for (int i = 0; i < 784; ++i) {
            input(0, i) = 0.5f;  
        }

        std::cout << "\n=== 执行 Forward ===" << std::endl;
        Matrix<float> output = model->forward(input);

        std::cout << "\n=== 输出结果 ===" << std::endl;
        std::cout << "维度：" << output.rows << " * " << output.cols << std::endl;
        output.print();

        float sum = 0.0f;
        float max_val = output(0, 0);
        int max_idx = 0;
        for (int j = 0; j < output.cols; ++j) {
            sum += output(0, j);
            if (output(0, j) > max_val) {
                max_val = output(0, j);
                max_idx = j;
            }
        }
        
        std::cout << "\n=== 验证 ===" << std::endl;
        std::cout << "Softmax 和：" << sum << std::endl;
        std::cout << "预测类别：" << max_idx << " (概率：" << max_val << ")" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误：" << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}