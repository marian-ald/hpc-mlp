
#include <iostream> 
#include <vector> 
#include <stdio.h>
#include <stdlib.h>

#include "utils.hpp"



/* Class implementing the multilayer perceptron */
class MLP {
private:
    int aNumberOfLayers;
    int aFeatureSize, aOutputSize;
    int aEpochs;
    double aLearningRate;
    double*** aCorrectWeights;
    double*** aWeights;
    double** aLocalGradients;
    double** aLayersInputs;
    double** aLayersOutputs;
    double** aLayersDerivatives;
    
    std::vector<int> aLayersDim;

    /* Private methods */
    void propagateForward(double* sample);
    void propagateBackward(double* sample, double label);

    void allocateWeights(double**** weights);
    void initWeights();
    void matrixMultiply(double* mat1, double** mat2, double* result, int size_mat1, int size_mat2);
    void sigmoidActivation(double* input, double* output, int length);

    void computeGradients(int layer_number, double* expected_output);
    void sigmoidDerivative(double* output, double* derivative, int length);


public:

    MLP(std::vector<int> topology, int epochs, double learningRate = 0.005);
    void train(dataset_t);
    std::vector<double> test(dataset_t* test_set);
    ~MLP();    
};
