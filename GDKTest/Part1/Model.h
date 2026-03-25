#ifndef MODEL_H
#define MODEL_H

#include "Matrix.h"

class Model {
public:
    Matrix weight1;  
    Matrix bias1;    
    Matrix weight2;  
    Matrix bias2;   

    Model(Matrix w1, Matrix b1, Matrix w2, Matrix b2)
        : weight1(w1), bias1(b1), weight2(w2), bias2(b2) {}

    Matrix forward(const Matrix& input) {
        Matrix temp1 = input * weight1;
        Matrix temp2 = temp1 + bias1;
        Matrix temp3 = relu(temp2);
        Matrix temp4 = temp3 * weight2;
        Matrix temp5 = temp4 + bias2;
        Matrix output = softmax(temp5);
        return output;
    }
};

#endif