#include "trainer.hpp"
#include "common.hpp"
#include <iostream>

vector<vector<double>> Trainer::computeDeltas(NeuralNetwork& nn, const vector<double>& target, ActivationType activation){
	int num_layers = nn.layers.size();
	vector<vector<double>> deltas(num_layers); 

	int L = num_layers -1; 
	int num_neurons_L = nn.layers[L].neurons.size(); 
	deltas[L].resize(num_neurons_L);

	for (int j=0; j < num_neurons; ++j){
		double x_j_L = nn.layers[L].last_outputs[j]; //¿Puede acceder a esta variable?
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


void Trainer::train(
    NeuralNetwork& nn,
    const vector<vector<double>>& data,
		const vector<vector<double>>& targets,
    int epochs,
    ActivationType activation)
{
	for (int e=0; e < epochs; ++e){
		double total_error = 0.0;

		for (size_t i=0; i < data_size(); ++i){
			nn.predict(data[i], activation);
			vector<vector<double>> deltas = computeDeltas(nn, targets[i], activation);

			applyGradients(nn, deltas, this->learning_rate); 
		}
	}
}


bool Trainer::train(
    Perceptron& perceptron,
    const vector<vector<double>>& training_inputs,
    const vector<double>& training_outputs,
    double learning_rate,
    int max_epochs,
    ActivationType activation)
{
    if (training_inputs.empty() || training_inputs.size() != training_outputs.size()) {
        cerr << "Error: Invalid training data\n";
        return false;
    }

    int input_size = training_inputs[0].size();
    if (perceptron.weights.size() != input_size) {
        cerr << "Error: Perceptron input size doesn't match training data\n";
        return false;
    }

    cout << "Starting Perceptron Learning Algorithm...\n";
    cout << "Learning rate: " << learning_rate << "\n";
    cout << "Max epochs: " << max_epochs << "\n";
    cout << "Training samples: " << training_inputs.size() << "\n";
    
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        int errors = 0;
        
        for (size_t i = 0; i < training_inputs.size(); ++i) {
            double prediction = perceptron.predict(training_inputs[i], activation);
            double target = training_outputs[i];
            double error = target - prediction;
            if (abs(error) > 0.5) {
                errors++;
                for (int j = 0; j < input_size; ++j) {
                    perceptron.weights[j] += learning_rate * error * training_inputs[i][j];
                }
                perceptron.bias += learning_rate * error;
            }
        }

        if (epoch % 10 == 0 || errors == 0) {
            cout << "Epoch " << epoch << ": " << errors << " errors\n";
        }
        
        if (errors == 0) {
            cout << "\nConverged after " << epoch + 1 << " epochs!\n";
            int correct = 0;
            for (size_t i = 0; i < training_inputs.size(); ++i) {
                double prediction = perceptron.predict(training_inputs[i], activation);
                double target = training_outputs[i];
                if ((prediction >= 0 && target >= 0) || (prediction < 0 && target < 0)) {
                    correct++;
                }
            }
            
            double accuracy = 100.0 * correct / training_inputs.size();
            cout << "Training accuracy: " << accuracy << "%\n";
            return true;
        }
    }

    cout << "\nFailed to converge after " << max_epochs << " epochs\n";
    int correct = 0;
    for (size_t i = 0; i < training_inputs.size(); ++i) {
        double prediction = perceptron.predict(training_inputs[i], activation);
        double target = training_outputs[i];
        
        if ((prediction >= 0 && target >= 0) || (prediction < 0 && target < 0)) {
            correct++;
        }
    }
    
    double accuracy = 100.0 * correct / training_inputs.size();
    cout << "Final training accuracy: " << accuracy << "%\n";
    
    return accuracy > 95.0;
}


double Trainer::test_acc(
    NeuralNetwork& network,
    const vector<vector<double>>& test_inputs,
    const vector<double>& test_outputs)
{
    if (network.layers.empty() || test_inputs.size() != test_outputs.size()) {
        return 0.0;
    }

    int correct = 0;
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        vector<double> output = network.predict(test_inputs[i]);
        if (output.size() == 1) {
            double prediction = output[0];
            double target = test_outputs[i];
            if (network.activation_type == ActivationType::STEP) {
                if ((prediction >= 0 && target >= 0) || (prediction < 0 && target < 0)) {
                    correct++;
                }
            } else {
                if ((prediction >= 0.5 && target >= 0.5) || (prediction < 0.5 && target < 0.5)) {
                    correct++;
                }
            }
        }
    }

    return 100.0 * correct / test_inputs.size();
}
