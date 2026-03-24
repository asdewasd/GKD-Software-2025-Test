#include "Part2.h"

using json = nlohmann::json;

int main() {
    std::ifstream meta_file("C:/Users/13298/Cprogram/GDKTest/Part2/mnist-fc/meta.json");
    if (!meta_file) {
        std::cerr << "无法打开 meta.json" << std::endl;
        return 1;
    }
    json meta;
    meta_file >> meta;

    auto fc1_weight_shape = meta["fc1.weight"].get<std::vector<int>>();
    auto fc1_bias_shape   = meta["fc1.bias"].get<std::vector<int>>();
    auto fc2_weight_shape = meta["fc2.weight"].get<std::vector<int>>();
    auto fc2_bias_shape   = meta["fc2.bias"].get<std::vector<int>>();

    int w1_rows = fc1_weight_shape[0], w1_cols = fc1_weight_shape[1];
    int b1_rows = fc1_bias_shape[0],   b1_cols = fc1_bias_shape[1];
    int w2_rows = fc2_weight_shape[0], w2_cols = fc2_weight_shape[1];
    int b2_rows = fc2_bias_shape[0],   b2_cols = fc2_bias_shape[1];

    Matrix w1 = read_binary_matrix("Part2/mnist-fc/fc1.weight", w1_rows, w1_cols);
    Matrix b1 = read_binary_matrix("Part2/mnist-fc/fc1.bias",   b1_rows, b1_cols);
    Matrix w2 = read_binary_matrix("Part2/mnist-fc/fc2.weight", w2_rows, w2_cols);
    Matrix b2 = read_binary_matrix("Part2/mnist-fc/fc2.bias",   b2_rows, b2_cols);

    Model model(w1, b1, w2, b2);

    Matrix x(1, 784);
    Matrix y = model.forward(x);
    y.printfMatrix();

    return 0;
}