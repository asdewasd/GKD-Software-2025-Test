#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cstring>
#include <cmath>
using namespace std;

class Matrix{
    public:
        //构造器
        Matrix(int rows,int cols){
            this->rows=rows;
            this->cols=cols;
            data=new float[rows*cols]();
        }
        Matrix() : rows(0), cols(0), data(nullptr) {}
        //析构函数（自动销毁）
        ~Matrix(){
            delete []data;
        }
        //拷贝函数
        Matrix(const Matrix &other){
            this->rows=other.rows;
            this->cols=other.cols;
            this->data=new float[rows*cols];
            memcpy(data,other.data,rows*cols*sizeof(float));
        }
        //重载赋值运算
        Matrix& operator=(const Matrix &other){
            if(this==&other)return *this;
            delete []data;
            this->rows=other.rows;
            this->cols=other.cols;
            this->data=new float[rows*cols];
            memcpy(data, other.data, rows*cols*sizeof(float)); 
            return *this;
        }
        //访问元素（通过坐标）
        float& operator()(int i, int j) {
            return data[i * cols + j];
        }
        //重载矩阵加法
       Matrix operator+(const Matrix &other) const {
            if (rows != other.rows || cols != other.cols) {
                cerr << "矩阵维度不同，无法相加" << endl;
                return *this; 
            }

            Matrix res(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    res(i, j) = data[i * cols + j] + other.data[i * cols + j];
                }
            }
            return res;
        }
        //重载矩阵乘法
        Matrix operator*(const Matrix &other)const{
            if(cols!=other.cols){
                cerr<<"矩阵维度不同，无法相乘"<<endl;
                return *this;
            }
            Matrix res(rows,other.cols);
            for(int i=0;i<rows;i++){
                for(int j=0;j<other.cols;j++){
                    float sum=0;
                    for(int k=0;k<cols;k++){
                        sum+=data[i*cols+k]*other.data[k*other.cols+j];
                    }
                    res(i,j)=sum;
                }
            }
            return res;
        }
        //relu函数
        Matrix relu()const{
            Matrix res(rows,cols);
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    float val=data[i*cols+j];
                    res(i,j)=(val>0?val:0.0f);
                }
            }
            return res;
        }
        //softMax
        Matrix softmax()const{
            if(rows!=1 && cols!=1){
                cerr<<"矩阵维度错误"<<endl;
                return *this;
            }
            int len=(rows==1?cols:rows);
            float max_val=data[0];
            for(int i=0;i<len;++i){
                if(data[i]>max_val){
                    max_val=data[i];
                }
            }
            float sum_exp=0.0f;
            for(int i=0;i<len;++i){
                sum_exp+=exp(data[i]-max_val);
            }
            Matrix res(rows,cols);
            for(int i=0;i<len;++i){
                res.data[i]=exp(data[i]-max_val)/sum_exp;
            }
            return res;
        }
        void printfMatrix()const{
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    cout<<data[i*cols+j]<<"\t";
                }
                cout << endl;
            }
        }

        int getRows() const{return rows;}   

        int getCols() const{return cols;}
    private:
        int rows,cols;
        float *data;
};

class Model {
public:
    Matrix weight1;
    Matrix bias1;
    Matrix weight2;
    Matrix bias2;
    // 构造函数：传入四个矩阵
    Model(const Matrix& w1, const Matrix& b1, const Matrix& w2, const Matrix& b2)
        : weight1(w1), bias1(b1), weight2(w2), bias2(b2) {}

    Matrix forward(const Matrix& input) {
        Matrix temp1 = input * weight1;
        Matrix temp2 = temp1 + bias1;
        Matrix temp3 = temp2.relu();
        Matrix temp4 = temp3 * weight2;
        Matrix temp5 = temp4 + bias2;
        Matrix output = temp5.softmax();
        return output;
    }
};

#endif