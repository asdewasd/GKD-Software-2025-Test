#include "Part1.h"
using namespace std;

int main() {
    Matrix vec_row(1, 3);
    vec_row(0, 0) = 2.0f;
    vec_row(0, 1) = 1.0f;
    vec_row(0, 2) = 0.0f;

    cout << "原行向量：" << endl;
    vec_row.printfMatrix();
    Matrix sm_row = vec_row.softmax();
    cout << "\nSoftmax 结果：" << endl;
    sm_row.printfMatrix();

    Matrix vec_col(3, 1);
    vec_col(0, 0) = 2.0f;
    vec_col(1, 0) = 1.0f;
    vec_col(2, 0) = 0.0f;

    cout << "\n原列向量：" << endl;
    vec_col.printfMatrix();
    Matrix sm_col = vec_col.softmax();
    cout << "\nSoftmax 结果：" << endl;
    sm_col.printfMatrix();

    return 0;
}