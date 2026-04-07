#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

template<typename T>
class Matrix {
public:
    int rows;
    int cols;
    T* data;

    Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
        data = new T[rows * cols]();
    }

    Matrix(const Matrix& other) {
        rows = other.rows;
        cols = other.cols;
        data = new T[rows * cols];
        memcpy(data, other.data, rows * cols * sizeof(T));
    }

    ~Matrix() {
        delete[] data;
    }

    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;
        delete[] data;
        rows = other.rows;
        cols = other.cols;
        data = new T[rows * cols];
        memcpy(data, other.data, rows * cols * sizeof(T));
        return *this;
    }

    T& operator()(int i, int j) {
        return data[i * cols + j];
    }

    const T& operator()(int i, int j) const {
        return data[i * cols + j];
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::runtime_error("矩阵乘法：尺寸不匹配！");
        }

        Matrix res(rows, other.cols);

        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;

        std::vector<std::thread> threads;

        auto worker = [&](int start_row, int end_row) {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < other.cols; ++j) {
                    T sum = T{};
                    for (int k = 0; k < cols; ++k) {
                        sum += data[i * cols + k] *
                               other.data[k * other.cols + j];
                    }
                    res(i, j) = sum;
                }
            }
        };

        int chunk = rows / num_threads;
        int remainder = rows % num_threads;

        int current = 0;

        for (int t = 0; t < num_threads; ++t) {
            int start = current;
            int end = start + chunk + (t < remainder ? 1 : 0);
            current = end;

            threads.emplace_back(worker, start, end);
        }

        for (auto& th : threads) th.join();

        return res;
    }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::runtime_error("矩阵加法：尺寸不匹配！");
        }
        Matrix res(rows, cols);
        for (int i = 0; i < rows * cols; ++i) {
            res.data[i] = data[i] + other.data[i];
        }
        return res;
    }

    void print() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << data[i * cols + j] << "\t";
            }
            std::cout << std::endl;
        }
    }

    static Matrix loadFromFile(const std::string& filepath, int rows, int cols) {
        Matrix matrix(rows, cols);
        std::ifstream file(filepath, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("无法打开文件：" + filepath);
        }

        file.read(reinterpret_cast<char*>(matrix.data),
                  rows * cols * sizeof(T));

        if (file.gcount() != rows * cols * sizeof(T)) {
            throw std::runtime_error("文件读取不完整：" + filepath);
        }

        file.close();
        return matrix;
    }
};

// ReLU
template<typename T>
Matrix<T> relu(const Matrix<T>& mat) {
    Matrix<T> res(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows * mat.cols; ++i) {
        res.data[i] = std::max(mat.data[i], T{0});
    }
    return res;
}

// Softmax
template<typename T>
Matrix<T> softmax(const Matrix<T>& vec) {
    if (vec.rows != 1 && vec.cols != 1) {
        throw std::runtime_error("SoftMax：输入必须是向量！");
    }
    Matrix<T> res(vec.rows, vec.cols);
    int n = vec.rows * vec.cols;

    T max_val = vec.data[0];
    for (int i = 1; i < n; ++i) {
        if (vec.data[i] > max_val) max_val = vec.data[i];
    }

    T sum_exp = T{};
    for (int i = 0; i < n; ++i) {
        sum_exp += std::exp(vec.data[i] - max_val);
    }

    for (int i = 0; i < n; ++i) {
        res.data[i] = std::exp(vec.data[i] - max_val) / sum_exp;
    }

    return res;
}

#endif