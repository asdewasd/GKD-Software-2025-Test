#ifndef MODEL_H
#define MODEL_H

#include "Matrix.h"
#include "ModelBase.h"

template<typename T>
class Model : public ModelBase {
public:
    Matrix<T> weight1;  
    Matrix<T> bias1;    
    Matrix<T> weight2;  
    Matrix<T> bias2;   

    Model(Matrix<T> w1, Matrix<T> b1, Matrix<T> w2, Matrix<T> b2)
        : weight1(w1), bias1(b1), weight2(w2), bias2(b2) {}

    Matrix<T> forwardTyped(const Matrix<T>& input) {
        Matrix<T> temp1 = input * weight1;
        Matrix<T> temp2 = temp1 + bias1;
        Matrix<T> temp3 = relu(temp2);
        Matrix<T> temp4 = temp3 * weight2;
        Matrix<T> temp5 = temp4 + bias2;
        Matrix<T> output = softmax(temp5);
        return output;
    }

    Matrix<float> forward(const Matrix<float>& input) override {
        Matrix<T> inputT(input.rows, input.cols);
        for (int i = 0; i < input.rows * input.cols; ++i) {
            inputT.data[i] = static_cast<T>(input.data[i]);
        }

        Matrix<T> outputT = forwardTyped(inputT);

        Matrix<float> output(outputT.rows, outputT.cols);
        for (int i = 0; i < outputT.rows * outputT.cols; ++i) {
            output.data[i] = static_cast<float>(outputT.data[i]);
        }
        return output;
    }

    std::string getModelType() const override {
        return typeid(T).name();
    }

    void printShape() const override {
        std::cout << "weight1: " << weight1.rows << " * " << weight1.cols << std::endl;
        std::cout << "bias1: " << bias1.rows << " * " << bias1.cols << std::endl;
        std::cout << "weight2: " << weight2.rows << " * " << weight2.cols << std::endl;
        std::cout << "bias2: " << bias2.rows << " * " << bias2.cols << std::endl;
    }
};

#endif