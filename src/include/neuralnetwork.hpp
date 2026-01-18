#ifndef NEURALNETWORK_INTERFACE_HPP
#define NEURALNETWORK_INTERFACE_HPP

#include <vector>
#include <string>
#include <map>
#include <set>

using std::vector;
using std::string;
using std::map;
using std::set;

enum class ActivationType { SIGMOID, TANH, RELU, STEP };

struct Dataset{
    vector<vector<double>> train_inputs;
    vector<int> train_targets;
    vector<vector<double>> val_inputs;
    vector<int> val_targets;
    int num_classes;
    map<int, string> id_to_label;
};


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
  int predict(const vector<double>& inputs);
  int predicted_class() const;
  vector<double> raw_output() const;
};


class Trainer{
private:
    double learning_rate;

    vector<vector<double>> computeDeltas(NeuralNetwork& nn, int target_class) const;  // Changed to int
    void applyGradients(NeuralNetwork& nn, const vector<vector<double>>& deltas, double eta) const;

public:
    Trainer(double lr = 0.1) : learning_rate(lr){}

    double train(
        NeuralNetwork& nn,
        const Dataset& ds,
        const int epochs) const;

    double train(
        Perceptron& perceptron,
        const vector<vector<double>>& training_inputs,
        const vector<int>& training_outputs,
        double learning_rate = 0.1,
        int max_epochs = 1000,
        ActivationType activation = ActivationType::STEP) const;
    
    double test_acc(
        NeuralNetwork& network,
        const vector<vector<double>>& test_inputs,
        const vector<int>& test_outputs) const;
};

Dataset loadDataset(const std::string& filename, size_t target_col, char delimiter, bool has_header, double val_split);

#endif
