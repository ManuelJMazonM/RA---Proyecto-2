#ifndef NEURALNETWORK_INTERFACE_HPP
#define NEURALNETWORK_INTERFACE_HPP

#include <vector>

using std::vector;

enum class ActivationType { SIGMOID, TANH, RELU, STEP };


class Perceptron {
public:
    vector<double> weights;
    double bias;

    Perceptron(int input_size);
    double predict(const vector<double>& inputs, ActivationType activation);
};


class Layer {
public:
    vector<Perceptron> neurons;
    vector<double> last_inputs;
    vector<double> last_outputs;

    Layer(int num_neurons, int inputs_per_neuron);
    vector<double> forward(const vector<double>& inputs, const ActivationType activation);
};


class NeuralNetwork {
public:
  vector<Layer> layers;
  ActivationType activation_type;

  NeuralNetwork(const vector<int>& topology, const ActivationType activation_type);
  vector<double> predict(const vector<double>& inputs);
};


class Trainer{
private:
	double learning_rate;

	vector<vector<double>> computeDeltas(NeuralNetwork& nn, const vector<double>& target);
	void applyGradients(NeuralNetwork& nn, const vector<vector<double>>& deltas, double eta);

public:
	Trainer(double lr = 0.1) : learning_rate(lr){}

	void train(
      NeuralNetwork& nn,
      const vector<vector<double>>& data,
      const vector<vector<double>>& targets,
      const int epochs); 

  bool train(
      Perceptron& perceptron,
      const vector<vector<double>>& training_inputs,
      const vector<double>& training_outputs,
      double learning_rate = 0.1,
      int max_epochs = 20,
      ActivationType activation = ActivationType::STEP);
  
  double test_acc(
      NeuralNetwork& network,
      const vector<vector<double>>& test_inputs,
      const vector<double>& test_outputs);
};

#endif
