#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cstring>
#include <cmath>

class Matrix {
public:
    int rows;   
    int cols;    
    float* data; 

    Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
        data = new float[rows * cols](); 
    }

    Matrix(const Matrix& other) {
        rows = other.rows;
        cols = other.cols;
        data = new float[rows * cols];
        memcpy(data, other.data, rows * cols * sizeof(float));
    }

    ~Matrix() {
        delete[] data;
    }

    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this; 

        delete[] data;

        rows = other.rows;
        cols = other.cols;
        data = new float[rows * cols];
        memcpy(data, other.data, rows * cols * sizeof(float));
        return *this;
    }

    float& operator()(int i, int j) {
        return data[i * cols + j]; 
    }

    const float& operator()(int i, int j) const {
        return data[i * cols + j];
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

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::runtime_error("矩阵乘法：尺寸不匹配（左列数≠右行数）！");
        }

        Matrix res(rows, other.cols);
        for (int i = 0; i < rows; ++i) {        
            for (int j = 0; j < other.cols; ++j) { 
                float sum = 0.0f;
                for (int k = 0; k < cols; ++k) {   
                    sum += data[i * cols + k] * other.data[k * other.cols + j];
                }
                res(i, j) = sum;
            }
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
};
Matrix relu(const Matrix& mat) {
    Matrix res(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows * mat.cols; ++i) {
        res.data[i] = std::max(mat.data[i], 0.0f); 
    }
    return res;
}

Matrix softmax(const Matrix& vec) {
    if (vec.rows != 1 && vec.cols != 1) {
        throw std::runtime_error("SoftMax：输入必须是向量（行/列=1）！");
    }

    Matrix res(vec.rows, vec.cols);
    int n = vec.rows * vec.cols; 
    float max_val = vec.data[0];

    for (int i = 1; i < n; ++i) {
        if (vec.data[i] > max_val) {
            max_val = vec.data[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_exp += std::exp(vec.data[i] - max_val);
    }

    for (int i = 0; i < n; ++i) {
        res.data[i] = std::exp(vec.data[i] - max_val) / sum_exp;
    }

    return res;
}

#endif 