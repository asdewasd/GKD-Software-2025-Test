#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Matrix.h"
#include "Model.h"

using namespace std;

Matrix<float> imageToMatrix(const cv::Mat& img) {
    Matrix<float> input(1, 784);

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            int idx = i * 28 + j;  // 行优先

            float pixel = img.at<uchar>(i, j);

            // 归一化到 0~1
            input(0, idx) = pixel / 255.0f;
        }
    }
    return input;
}

int argmax(const Matrix<float>& output) {
    int pred = 0;
    float max_val = output(0, 0);

    for (int i = 1; i < 10; i++) {
        if (output(0, i) > max_val) {
            max_val = output(0, i);
            pred = i;
        }
    }
    return pred;
}

int main() {

    string folderPath = "mnist-fc-plus";  

    vector<int> w1_shape = {784, 128};
    vector<int> b1_shape = {1, 128};
    vector<int> w2_shape = {128, 10};
    vector<int> b2_shape = {1, 10};

    Matrix<float> w1 = Matrix<float>::loadFromFile(folderPath + "/fc1.weight", w1_shape[0], w1_shape[1]);
    Matrix<float> b1 = Matrix<float>::loadFromFile(folderPath + "/fc1.bias", b1_shape[0], b1_shape[1]);
    Matrix<float> w2 = Matrix<float>::loadFromFile(folderPath + "/fc2.weight", w2_shape[0], w2_shape[1]);
    Matrix<float> b2 = Matrix<float>::loadFromFile(folderPath + "/fc2.bias", b2_shape[0], b2_shape[1]);

    Model<float> model(w1, b1, w2, b2);

    // ====== 遍历 nums 文件夹 ======
    for (int digit = 0; digit <= 9; digit++) {

        string path = "nums/" + to_string(digit) + ".png";

        // 1️⃣ 读取灰度图
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

        if (img.empty()) {
            cout << "读取失败: " << path << endl;
            continue;
        }

        // 2️⃣ resize 到 28×28
        cv::resize(img, img, cv::Size(28, 28));

        // ⚠️ 如果你的图片是黑底白字，取消注释
        // cv::bitwise_not(img, img);

        // 3️⃣ 转 Matrix
        Matrix<float> input = imageToMatrix(img);

        // 4️⃣ forward
        Matrix<float> output = model.forward(input);

        // 5️⃣ 预测
        int pred = argmax(output);

        cout << "真实: " << digit << " 预测: " << pred << endl;
    }

    return 0;
}