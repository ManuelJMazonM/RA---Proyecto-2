#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <cmath>

enum class ActivationType { SIGMOID, TANH, RELU, STEP };

namespace Activations {
    double apply(double s, ActivationType type);
    double applyDerivate(double f_s, ActivationType type);
}

<<<<<<< Updated upstream
#endif
=======
struct Dataset{
    std::vector<std::vector<double>> train_inputs;
    std::vector<std::vector<double>> train_targets;
    std::vector<std::vector<double>> val_inputs;
    std::vector<std::vector<double>> val_targets;
    int num_classes;
}

#endif
>>>>>>> Stashed changes
