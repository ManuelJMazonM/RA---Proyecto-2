#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <set>

enum class ActivationType { SIGMOID, TANH, RELU, STEP };

namespace Activations {
    double apply(double s, ActivationType type);
    double applyDerivative(double f_s, ActivationType type);
}

struct Dataset{
    std::vector<std::vector<double>> train_inputs;
    std::vector<int> train_targets;
    std::vector<std::vector<double>> val_inputs;
    std::vector<int> val_targets;
    int num_classes;
    std::map<int, std::string> id_to_label;
};

Dataset loadDataset(const std::string& filename, size_t target_col, char delimiter, bool has_header, double val_split);

#endif
