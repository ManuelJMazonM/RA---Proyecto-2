#include "Common.hpp"
#include <algorithm>

namespace Activations {
	//Funciones de Activación
	double apply(double s, ActivationType type){
		switch (type){
		case ActivationType::SIGMOID:
			return 1.0 / (1.0 + std::exp(-s));
		case ActivationType::TANH:
			return std::tanh(s);
		case ActivationType::RELU:
			return std::max(0.0, s); 
		case ActivationType::STEP:
			return (s >= 0.0) ? 1.0 : -1.0;
		default: 
			return s;
		}
	}


	//Derivadas
	double applyDerivate(double f_s, ActivationType type){
		switch (type){
		case ActivationType::SIGMOID:
			return f_s * (1.0 - f_s);
		case ActivationType::TANH:
			return 1.0 - (f_s * f_s);
		case ActivationType::RELU:
			return (f_s > 0.0) ? 1.0 : 0.0;
		case ActivationType::STEP:
			return 0.0; //NO se usa en BackProp, así que no necesitamos la derivada
		default: 
			return 1.0;
		}	
	}
}