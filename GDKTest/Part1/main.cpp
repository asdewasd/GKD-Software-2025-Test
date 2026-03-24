#include <iostream>
#include "Matrix.h"
#include "Model.h"

int main() {
    try {
        // 1. 创建空白矩阵（Part2前用空白矩阵测试）
        Matrix weight1(784, 500); // 784×500空白矩阵
        Matrix bias1(1, 500);     // 1×500空白矩阵
        Matrix weight2(500, 10);  // 500×10空白矩阵
        Matrix bias2(1, 10);      // 1×10空白矩阵

        // 2. 创建Model对象
        Model model(weight1, bias1, weight2, bias2);

        // 3. 构造测试输入（1×784矩阵，元素全为1.0f）
        Matrix input(1, 784);
        for (int j = 0; j < 784; ++j) {
            input(0, j) = 1.0f;
        }

        // 4. 调用forward方法
        Matrix output = model.forward(input);

        // 5. 打印输出（1×10概率向量）
        std::cout << "Model forward输出（1×10概率向量）：" << std::endl;
        output.print();

        // 验证SoftMax：所有元素和≈1
        float sum = 0.0f;
        for (int j = 0; j < 10; ++j) {
            sum += output(0, j);
        }
        std::cout << "\nSoftMax总和：" << sum << "（期望≈1）" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误：" << e.what() << std::endl;
        return -1;
    }

    return 0;
}