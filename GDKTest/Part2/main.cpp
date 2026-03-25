#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Matrix.h"
#include "Model.h"

using json = nlohmann::json;

// 从 JSON 解析矩阵维度
std::vector<int> getMatrixShape(const json& meta, const std::string& key) {
    return meta[key].get<std::vector<int>>();
}

// 加载模型
Model loadModel(const std::string& folderPath) {
    // 1. 读取 meta.json
    std::string jsonPath = folderPath + "/meta.json";
    std::ifstream meta_file(jsonPath);
    
    if (!meta_file.is_open()) {
        throw std::runtime_error("无法打开 meta.json: " + jsonPath);
    }
    
    json meta;
    meta_file >> meta;
    meta_file.close();
    
    std::cout << "=== JSON 解析结果 ===" << std::endl;
    std::cout << "type: " << meta["type"] << std::endl;
    
    // 2. 获取各矩阵维度
    auto w1_shape = getMatrixShape(meta, "fc1.weight");
    auto b1_shape = getMatrixShape(meta, "fc1.bias");
    auto w2_shape = getMatrixShape(meta, "fc2.weight");
    auto b2_shape = getMatrixShape(meta, "fc2.bias");
    
    std::cout << "fc1.weight: [" << w1_shape[0] << ", " << w1_shape[1] << "]" << std::endl;
    std::cout << "fc1.bias:   [" << b1_shape[0] << ", " << b1_shape[1] << "]" << std::endl;
    std::cout << "fc2.weight: [" << w2_shape[0] << ", " << w2_shape[1] << "]" << std::endl;
    std::cout << "fc2.bias:   [" << b2_shape[0] << ", " << b2_shape[1] << "]" << std::endl;
    
    // 3. 加载二进制矩阵文件
    std::cout << "\n=== 加载二进制文件 ===" << std::endl;
    
    Matrix w1 = Matrix::loadFromFile(folderPath + "/fc1.weight", w1_shape[0], w1_shape[1]);
    std::cout << "✓ fc1.weight 加载完成 (" << w1_shape[0] << " * " << w1_shape[1] << ")" << std::endl;
    
    Matrix b1 = Matrix::loadFromFile(folderPath + "/fc1.bias", b1_shape[0], b1_shape[1]);
    std::cout << "✓ fc1.bias 加载完成 (" << b1_shape[0] << " * " << b1_shape[1] << ")" << std::endl;
    
    Matrix w2 = Matrix::loadFromFile(folderPath + "/fc2.weight", w2_shape[0], w2_shape[1]);
    std::cout << "✓ fc2.weight 加载完成 (" << w2_shape[0] << " * " << w2_shape[1] << ")" << std::endl;
    
    Matrix b2 = Matrix::loadFromFile(folderPath + "/fc2.bias", b2_shape[0], b2_shape[1]);
    std::cout << "✓ fc2.bias 加载完成 (" << b2_shape[0] << " * " << b2_shape[1] << ")" << std::endl;
    
    return Model(w1, b1, w2, b2);
}

int main() {
    try {
        std::cout << "=== Part2 文件读取测试 ===" << std::endl;
        std::cout << std::endl;
        
        // 加载模型
        Model model = loadModel("mnist-fc");
        
        std::cout << "\n=== Model 创建成功 ===" << std::endl;
        
        // 创建输入 1 * 784
        Matrix input(1, 784);
        
        // 填充测试数据（模拟图像像素）
        for (int i = 0; i < 784; ++i) {
            input(0, i) = 0.5f;
        }
        
        // 前向传播
        std::cout << "\n=== 执行 Forward ===" << std::endl;
        Matrix output = model.forward(input);
        
        // 输出结果
        std::cout << "\n=== 输出结果 ===" << std::endl;
        std::cout << "维度：" << output.rows << " * " << output.cols << std::endl;
        output.print();
        
        // 验证
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