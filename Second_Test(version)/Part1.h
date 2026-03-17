#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cstring>
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
        Matrix operator*(const Matrix &other){
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
                }
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

#endif