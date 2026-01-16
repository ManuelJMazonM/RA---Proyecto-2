#include "trainer.hpp"
#include "common.hpp"
#include <iostream>

using std::cout;
using std::cerr;
using std::abs;
using std::exp;
using std::max;


vector<vector<double>> Trainer::computeDeltas(NeuralNetwork& nn, const vector<double>& target) const
{
	int num_layers = nn.layers.size();
	vector<vector<double>> deltas(num_layers); 

	int L = num_layers -1;
	int num_neurons_L = nn.layers[L].neurons.size(); 
	deltas[L].resize(num_neurons_L);

	for (int j=0; j < num_neurons_L; ++j)
  {
		double x_j_L = nn.layers[L].last_outputs[j];
		double error_derivative = 2.0 * (x_j_L - target[j]);
		double theta_prime = Activations::applyDerivative(x_j_L, nn.activation_type);

		deltas[L][j] = theta_prime * error_derivative;
	}

	for (int l = L -1; l >= 0; l--)
  {
		int num_neurons_l = nn.layers[l].neurons.size();
		deltas[l].resize(num_neurons_l);

		for (int i=0; i < num_neurons_l; ++i){
			double sum_deltas_next = 0.0;

			for (size_t j=0; j < nn.layers[l+1].neurons.size(); ++j){
				sum_deltas_next += nn.layers[l+1].neurons[j].weights[i] * deltas[l+1][j];
			}

			double x_i_l = nn.layers[l].last_outputs[i];
			double theta_prime = Activations::applyDerivative(x_i_l, nn.activation_type);

			deltas[l][i] = theta_prime * sum_deltas_next;
		}
	}

	return deltas; 
}


void Trainer::applyGradients(NeuralNetwork& nn, const std::vector<std::vector<double>>& deltas, double eta) const
{

	for (size_t l=0; l < nn.layers.size(); ++l){
		Layer& layer = nn.layers[l];

		for (size_t j=0; j < layer.neurons.size(); ++j){
			Perceptron& neuron = layer.neurons[j];
			double delta_j = deltas[l][j];

			for (size_t k=0; k < neuron.weights.size(); ++k){
				double x_k = layer.last_inputs[k];

				neuron.weights[k] -= eta * delta_j * x_k;
			}

			neuron.bias -= eta * delta_j * 1.0;
		}
	}
}


double Trainer::train(
    NeuralNetwork& nn,
    const vector<vector<double>>& train_input,
		const vector<int>& train_output,
    const int epochs) const
{
  for (int e=0; e < epochs; ++e)
  {
    for (size_t i=0; i < train_input.size(); ++i)
    {
			nn.predict(train_input[i]);
			vector<vector<double>> deltas = computeDeltas(nn, train_output[i]);
			applyGradients(nn, deltas, learning_rate); 
    }
	}

  double hits = 0;
  for(size_t i=0; i < train_input.size(); i++)
  {
    if(nn.predict(train_input[i]) == train_output[i]) hits++;
  }
  return hits / (double)train_output.size(); // accuracy
}


double Trainer::train(
    Perceptron& perceptron,
    const vector<vector<double>>& training_inputs,
    const vector<double>& training_outputs,
    double learning_rate,
    int max_epochs,
    ActivationType activation) const
{
  if (training_inputs.empty() || training_inputs.size() != training_outputs.size()) {
    cerr << "Error: Invalid training data\n";
    return false;
  }

  size_t input_size = training_inputs[0].size();
  if (perceptron.weights.size() != input_size) {
    cerr << "Error: Perceptron input size doesn't match training data\n";
    return false;
  }

  for (int epoch = 0; epoch < max_epochs; ++epoch) {
    for (size_t i = 0; i < training_inputs.size(); ++i) {
      double prediction = perceptron.predict(training_inputs[i], activation);
      double target = training_outputs[i];
      double error = target - prediction;
      if (abs(error) > 0.5) {
        for (size_t j = 0; j < input_size; ++j) {
          perceptron.weights[j] += learning_rate * error * training_inputs[i][j];
        }
        perceptron.bias += learning_rate * error;
      }
    }
  }

  int correct = 0;
  for (size_t i = 0; i < training_inputs.size(); ++i) {
    double prediction = perceptron.predict(training_inputs[i], activation);
    double target = training_outputs[i];

    if ((prediction >= 0 && target >= 0) || (prediction < 0 && target < 0)) {
      correct++;
    }
  }

  double accuracy = 100.0 * correct / training_inputs.size();
  return accuracy;
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


// TODO: modify
// --------------------------------------------------------------------------------------------------------------

static double targetToDouble(int target_class, ActivationType activation_type, size_t output_size, size_t neuron_index = 0) {
    if (output_size == 1) {
        // Single output neuron
        if (activation_type == ActivationType::STEP) {
            return (target_class > 0) ? 1.0 : -1.0;
        } else if (activation_type == ActivationType::SIGMOID) {
            return (target_class > 0) ? 1.0 : 0.0;
        } else {
            return static_cast<double>(target_class);
        }
    } else {
        // Multiple output neurons: one-hot encoding
        return (neuron_index == static_cast<size_t>(target_class)) ? 1.0 : 0.0;
    }
}

vector<vector<double>> Trainer::computeDeltas(NeuralNetwork& nn, int target_class) const
{
    int num_layers = nn.layers.size();
    vector<vector<double>> deltas(num_layers); 

    int L = num_layers - 1;
    int num_neurons_L = nn.layers[L].neurons.size(); 
    deltas[L].resize(num_neurons_L);

    if (nn.activation_type == ActivationType::STEP) {
        cerr << "Warning: STEP activation has zero derivative. No backpropagation\n";
        return deltas;
    }

    for (int j = 0; j < num_neurons_L; ++j)
    {
        double x_j_L = nn.layers[L].last_outputs[j];
        double target = targetToDouble(target_class, nn.activation_type, num_neurons_L, j);
        double error_derivative = 2.0 * (x_j_L - target);
        double theta_prime = Activations::applyDerivative(x_j_L, nn.activation_type);

        deltas[L][j] = theta_prime * error_derivative;
    }

    for (int l = L - 1; l >= 0; l--)
    {
        int num_neurons_l = nn.layers[l].neurons.size();
        deltas[l].resize(num_neurons_l);

        for (int i = 0; i < num_neurons_l; ++i){
            double sum_deltas_next = 0.0;

            for (size_t j = 0; j < nn.layers[l+1].neurons.size(); ++j){
                sum_deltas_next += nn.layers[l+1].neurons[j].weights[i] * deltas[l+1][j];
            }

            double x_i_l = nn.layers[l].last_outputs[i];
            double theta_prime = Activations::applyDerivative(x_i_l, nn.activation_type);

            deltas[l][i] = theta_prime * sum_deltas_next;
        }
    }

    return deltas; 
}


void Trainer::applyGradients(NeuralNetwork& nn, const std::vector<std::vector<double>>& deltas, double eta) const
{
    for (size_t l = 0; l < nn.layers.size(); ++l){
        Layer& layer = nn.layers[l];

        for (size_t j = 0; j < layer.neurons.size(); ++j){
            Perceptron& neuron = layer.neurons[j];
            double delta_j = deltas[l][j];

            for (size_t k = 0; k < neuron.weights.size(); ++k){
                double x_k = layer.last_inputs[k];
                neuron.weights[k] -= eta * delta_j * x_k;
            }

            neuron.bias -= eta * delta_j;
        }
    }
}


double Trainer::train(
    NeuralNetwork& nn,
    const vector<vector<double>>& train_input,
    const vector<int>& train_output,
    const int epochs) const
{
    if (train_input.empty() || train_input.size() != train_output.size()) {
        cerr << "Error: Invalid training data\n";
        return 0.0;
    }

    // Check for STEP activation warning
    if (nn.activation_type == ActivationType::STEP) {
        cerr << "Warning: Training with STEP activation. Backpropagation may not converge.\n";
    }

    for (int e = 0; e < epochs; ++e)
    {
        for (size_t i = 0; i < train_input.size(); ++i)
        {
            nn.predict(train_input[i]);
            vector<vector<double>> deltas = computeDeltas(nn, train_output[i]);
            applyGradients(nn, deltas, learning_rate); 
        }
    }

    double hits = 0;
    for(size_t i = 0; i < train_input.size(); i++)
    {
        if(nn.predict(train_input[i]) == train_output[i]) hits++;
    }
    
    double accuracy = hits / static_cast<double>(train_output.size());
    cout << "Training Accuracy: " << (accuracy * 100.0) << "%\n";
    return accuracy;
}


double Trainer::train(
    Perceptron& perceptron,
    const vector<vector<double>>& training_inputs,
    const vector<int>& training_outputs,  // Changed to int
    double learning_rate,
    int max_epochs,
    ActivationType activation) const
{
    if (training_inputs.empty() || training_inputs.size() != training_outputs.size()) {
        cerr << "Error: Invalid training data\n";
        return 0.0;
    }

    size_t input_size = training_inputs[0].size();
    if (perceptron.weights.size() != input_size) {
        cerr << "Error: Perceptron input size doesn't match training data\n";
        return 0.0;
    }

    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        for (size_t i = 0; i < training_inputs.size(); ++i) {
            double prediction = perceptron.predict(training_inputs[i], activation);
            
            // Convert integer target to appropriate double value
            double target;
            if (activation == ActivationType::STEP) {
                target = (training_outputs[i] > 0) ? 1.0 : -1.0;
            } else if (activation == ActivationType::SIGMOID) {
                target = (training_outputs[i] > 0) ? 1.0 : 0.0;
            } else {
                target = static_cast<double>(training_outputs[i]);
            }
            
            double error = target - prediction;
            
            // Update if error is significant
            if (abs(error) > 0.01) {
                for (size_t j = 0; j < input_size; ++j) {
                    perceptron.weights[j] += learning_rate * error * training_inputs[i][j];
                }
                perceptron.bias += learning_rate * error;
            }
        }
    }

    // Calculate accuracy
    int correct = 0;
    for (size_t i = 0; i < training_inputs.size(); ++i) {
        double prediction = perceptron.predict(training_inputs[i], activation);
        int predicted_class;
        
        if (activation == ActivationType::STEP) {
            predicted_class = (prediction >= 0.0) ? 1 : -1;
        } else if (activation == ActivationType::SIGMOID) {
            predicted_class = (prediction >= 0.5) ? 1 : 0;
        } else {
            predicted_class = static_cast<int>(round(prediction));
        }
        
        if (predicted_class == training_outputs[i]) {
            correct++;
        }
    }

    double accuracy = 100.0 * correct / training_inputs.size();
    return accuracy;
}


double Trainer::test_acc(
    NeuralNetwork& network,
    const vector<vector<double>>& test_inputs,
    const vector<int>& test_outputs) const  // Changed to int
{
    if (network.layers.empty() || test_inputs.size() != test_outputs.size()) {
        return 0.0;
    }

    int correct = 0;
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        if (network.predict(test_inputs[i]) == test_outputs[i]) {
            correct++;
        }
    }

    return 100.0 * correct / test_inputs.size();
}
[file content end]
