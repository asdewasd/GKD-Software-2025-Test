#ifndef MODEL_H
#define MODEL_H

#include "Matrix.h"

class Model {
private:
    Matrix weight1; 
    Matrix bias1;   
    Matrix weight2; 
    Matrix bias2;   

public:
    Model(const Matrix& w1, const Matrix& b1, const Matrix& w2, const Matrix& b2)
        : weight1(w1), bias1(b1), weight2(w2), bias2(b2) {
        if (w1.rows != 784 || w1.cols != 500) {
            throw std::runtime_error("weight1尺寸错误：必须是784×500！");
        }
        if (b1.rows != 1 || b1.cols != 500) {
            throw std::runtime_error("bias1尺寸错误：必须是1×500！");
        }
        if (w2.rows != 500 || w2.cols != 10) {
            throw std::runtime_error("weight2尺寸错误：必须是500×10！");
        }
        if (b2.rows != 1 || b2.cols != 10) {
            throw std::runtime_error("bias2尺寸错误：必须是1×10！");
        }
    }

    Matrix forward(const Matrix& x) {
        if (x.rows != 1 || x.cols != 784) {
            throw std::runtime_error("输入矩阵尺寸错误：必须是1×784！");
        }

        Matrix z1 = x * weight1;    
        Matrix a1 = z1 + bias1;    
        Matrix h1 = relu(a1);       
        Matrix z2 = h1 * weight2;  
        Matrix a2 = z2 + bias2;     
        Matrix output = softmax(a2); 

        return output;
    }
};

#endif 