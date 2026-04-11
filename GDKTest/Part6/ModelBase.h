#ifndef MODEL_BASE_H
#define MODEL_BASE_H

#include "Matrix.h"
#include <memory>
#include <string>

class ModelBase {
public:
    virtual ~ModelBase() = default;

    virtual Matrix<float> forward(const Matrix<float>& input) = 0;

    virtual std::string getModelType() const = 0;

    virtual void printShape() const = 0;
};

#endif