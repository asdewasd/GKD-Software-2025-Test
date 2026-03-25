#include <iostream>
#include "Matrix.h"
#include "Model.h"

int main() {
    std::cout << "=== Part1 测试 ===" << std::endl;

    Matrix w1(784, 500);
    Matrix b1(1, 500);
    Matrix w2(500, 10);
    Matrix b2(1, 10);

    Model model(w1, b1, w2, b2);
    std::cout << "Model 创建成功" << std::endl;

    Matrix input(1, 784);
    std::cout << "Input 形状：1 * 784" << std::endl;

    Matrix output = model.forward(input);
    std::cout << "Forward 完成" << std::endl;

    std::cout << "\n=== 输出结果 ===" << std::endl;
    std::cout << "Output 形状：" << output.rows << " * " << output.cols << std::endl;
    output.print();

    float sum = 0.0f;
    float max_val = output(0, 0);
    for (int j = 0; j < output.cols; ++j) {
        sum += output(0, j);
        if (output(0, j) > max_val) {
            max_val = output(0, j);
        }
    }

    std::cout << "\n=== 验证 ===" << std::endl;
    std::cout << "Softmax 和：" << sum << " (应接近 1.0)" << std::endl;
    std::cout << "最大值：" << max_val << std::endl;
    std::cout << "最小值：" << output(0, 0) << std::endl;

    return 0;
}