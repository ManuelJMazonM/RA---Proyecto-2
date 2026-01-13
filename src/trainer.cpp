#include "trainer.hpp"
#include "common.hpp"
#include <iostream>

std::vector<std::vector<double>> Trainer::computeDeltas(NeuralNetwork& nn, const std::vector<double>& target, ActivationType activation){
	int num_layers = nn.layers.size();
	std::vector<std::vector<double>> deltas(num_layers); 

	int L = num_layers -1; 
	int num_neurons_L = nn.layers[L].neurons.size(); 
	deltas[L].resize(num_neurons_L);

	for (int j=0; j < num_neurons; ++j){
		double x_j_L = nn.layers[L].last_outputs[j]; //¿Puede acceder a esta variable?

		//Derivada del error cuadrático 
		double error_derivative = 2.0 * (x_j_L - target[j]);

		double theta_prime = Activations::applyDerivative(x_j_L, activation);

		deltas[L][j] = theta_prime * error_derivative;
	}

	for (int l = L -1; l >= 0; --l){
		int num_neurons_l = nn.layers[l].neurons.size();
		deltas[l].resize(num_neurons_l);

		for (int i=0; i < num_neurons_l; ++i){
			double sum_deltas_next = 0.0;

			for (int j=0; j < nn.layers[l+1].neurons.size(); ++j){
				sum_deltas_next += nn.layers[l+1].neurons[j].weights[i] * deltas[l+1][j];
			}

			double x_i_l = nn.layers[l].last_outputs[i];
			double theta_prime = Activations::applyDerivative(x_i_l, activation);

			deltas[l][i] = theta_prime * sum_deltas_next;
		}
	}

	return deltas; 
}


void Trainer::trainBackprop(NeuralNetwork& nn, const std::vector<std::vector<double>>& data,
							const std::vector<std::vector<double>>& targets, int epochs, ActivationType activation){

	for (int e=0; e < epochs; ++e){
		double total_error = 0.0;

		for (size_t i=0; i < data_size(); ++i){
			nn.predict(data[i], activation);

			std::vector<std::vector<double>> deltas = computeDeltas(nn, targets[i], activation);

			applyGradients(nn, deltas, this->learning_rate); 
		}
	}
}

