#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <cmath>

enum class ActivationType { SIGMOID, TANH, RELU, STEP };

namespace Activations {
    double apply(double s, ActivationType type);
    double applyDerivative(double f_s, ActivationType type);
}

#endif
