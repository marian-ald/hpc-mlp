#include <stdio.h>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "MLP.hpp"
#include "utils.hpp"



/*___________________________________Public methods_________________________________________*/

/* Constructor of the MLP class */
MLP::MLP(std::vector<int> hidden_layers_dim, int epochs, double learning_rate):
                                                        aEpochs(epochs),
                                                        aLearningRate(learning_rate)
{
    this->aNumberOfLayers = hidden_layers_dim.size();
    this->aLayersDim = hidden_layers_dim;

    srand(time(NULL));

    allocateWeights(&aWeights);
    allocateWeights(&aCorrectWeights);

    initWeights();

    /* Allocate memory for the:
        -inputs of the neurons
        -outputs of the neurons
        -local gradients
        -layer derivatives
    */
    aLayersInputs = (double**)calloc(aNumberOfLayers, sizeof(double*));
    aLayersOutputs = (double**)calloc(aNumberOfLayers, sizeof(double*));
    aLocalGradients = (double**)calloc(aNumberOfLayers, sizeof(double*));
    aLayersDerivatives = (double**)calloc(aNumberOfLayers, sizeof(double*));

    for (int i = 0; i < this->aNumberOfLayers; ++i)
    {
        aLayersInputs[i] = (double*)calloc(hidden_layers_dim[i], sizeof(double));

        /* For each output layer add an entry for the bias */
        aLayersOutputs[i] = (double*)calloc(hidden_layers_dim[i] + 1, sizeof(double));
        aLocalGradients[i] = (double*)calloc(hidden_layers_dim[i], sizeof(double));
        aLayersDerivatives[i] = (double*)calloc(hidden_layers_dim[i], sizeof(double));
    }
}


/* Start training the network on a dataset */
void MLP::train(dataset_t dataset)
{
    aFeatureSize = dataset.aFeatureSize;
    aOutputSize = dataset.aOutputSize;

    /* Repeat the training for a predefined number of iterations(epochs) */
    for (int i = 0; i < this->aEpochs; ++i)
    {
        for (int j = 0; j < dataset.aNbElements; ++j)
        {
            propagateForward(dataset.aData[j]);

            propagateBackward(dataset.aData[j], dataset.aLabels[j]);
        }
    }
}


/* Predict the labels of a set of unseen samples */
std::vector<double> MLP::test(dataset_t* test_set)
{
    std::vector<double> predictions;

    for (int i = 0; i < test_set->aNbElements; ++i)
    {
        propagateForward(test_set->aData[i]);

        /* Store the predicted label,  */
        predictions.push_back(aLayersOutputs[aNumberOfLayers - 1][1]);
    }
    return predictions;
}


/*___________________________________Private methods_________________________________________*/

/*  Randomly initialize the weights within an interval defined by the number
    of neurons per each layer
*/
void MLP::initWeights()
{
    double* epsilon = (double*)calloc(aNumberOfLayers - 1, sizeof(double));

    for (int i = 0; i < aNumberOfLayers - 1; ++i)
    {
        epsilon[i] = sqrt(6.0 / (aLayersDim[i] + aLayersDim[i + 1]));
    }

    /*  Each weights matrix of the i-th layer is initialized with random numbers in the interval:
        [-epsilon[i], epsilon[i]]
    
        Since the initialization of the weights is run a single time, the paralellelization of this
        region should not bring a (great) improvement.

        In the end the weieghts will fit the Xavier Normal Distribution.
    */
    for (int i = 0; i < aNumberOfLayers - 1; ++i)
    {
        for (int j = 0; j < aLayersDim[i] + 1; ++j)
        {
            for (int k = 0; k < aLayersDim[i + 1]; ++k)
            {
                aWeights[i][j][k] = -epsilon[i] + ((double)rand() / ((double)RAND_MAX / (2.0 * epsilon[i])));
                // std::cout << aWeights[i][j][k] << std::endl;
            }
        }
    }
    free(epsilon);
}


/*  Propagate a sample(1D array) through the network. 
    This is reduced to a succession of matrix multiplications until reaching
    the output layer.
*/
void MLP::propagateForward(double* sample)
{
    /* Initialize the bias of the feature layer with 1 */
    aLayersOutputs[0][0] = 1;

    /* The input and output of the first layer are exactly the feature vector */
    for (int i = 0; i < aFeatureSize; ++i)
    {
        aLayersOutputs[0][i + 1] = aLayersInputs[0][i] = sample[i];      
    }

    /* Compute the outputs of each hidden layer, then each if its activations */
    for (int i = 1; i < aNumberOfLayers - 1; ++i)
    {
        matrixMultiply(aLayersOutputs[i - 1], aWeights[i - 1], aLayersInputs[i], aLayersDim[i - 1] + 1, aLayersDim[i]);
        sigmoidActivation(aLayersInputs[i], aLayersOutputs[i], aLayersDim[i]);
    }
    
    /* Compute the activations of the output layer */
    matrixMultiply(aLayersOutputs[aNumberOfLayers - 2], aWeights[aNumberOfLayers - 2],
                    aLayersInputs[aNumberOfLayers - 1], aLayersDim[aNumberOfLayers - 2] + 1,
                    aLayersDim[aNumberOfLayers - 1]);
    sigmoidActivation(aLayersInputs[aNumberOfLayers - 1], aLayersOutputs[aNumberOfLayers - 1], aLayersDim[aNumberOfLayers - 1]);

}


/* Compute the error of a sample then back-propagate it through the network, from
    the last layer, to the first one.
*/
void MLP::propagateBackward(double* sample, double label)
{
    /* Store the expected label as an array of bits */
    double* expected_output = (double*)calloc(aOutputSize, sizeof(double));
    int i, j, k;
    /* Differentiate between bi-classification and multi-classification */
    if (aOutputSize == 1)
    {
        expected_output[0] = label;
    } else
    {
        expected_output[int(label) - 1] = 1;        
    }

    /* Compute the gradients for the output layer */
    computeGradients(aNumberOfLayers - 1, expected_output);

    /* Compute corrections for the output layer weights */
    for (i = 0; i < aOutputSize; ++i)
    {
        for (j = 0; j < aLayersDim[aNumberOfLayers - 2] + 1; ++j)
        {
            aCorrectWeights[aNumberOfLayers - 2][j][i] = aLearningRate * aLocalGradients[aNumberOfLayers - 1][i] * aLayersOutputs[aNumberOfLayers - 2][j];
        }
    }

    /*  Compute the corrections of the weigths, using the local gradient, the layer output
        and the learning rate */
    for (i = aNumberOfLayers - 2; i >= 1; --i)
    {
        computeGradients(i, expected_output);

        /* I will resume to only parallelize the second for loop, since in the previous called function
            <<computeGradients>>, there already exists a prallel region
        */
        #pragma omp parallel for shared(j) private(k)
        for (j = 0; j < aLayersDim[i]; ++j)
        {
            for (k = 0; k < aLayersDim[i - 1] + 1; ++k)
            {
                aCorrectWeights[i-1][k][j] = aLearningRate * aLocalGradients[i][j] * aLayersOutputs[i - 1][k];
            }
        }
    }

    /* Tune the weights by applying the corrections */
    // #pragma omp parallel for shared(i) private(j,k)
    for (i = 0; i < aNumberOfLayers - 1; ++i)
    {
        /* Another prallelization option would be to palce the pragma before the first for
            loop. After testing this, I observed the execution time was better if parallelizing the
            2nd loop.
        */
        #pragma omp parallel for shared(i,j) private(k)
        for (j = 0; j < aLayersDim[i] + 1; ++j)
        {
            for (k = 0; k < aLayersDim[i + 1]; ++k)
            {
                aWeights[i][j][k] -= aCorrectWeights[i][j][k];
            }
        }
    }
    free(expected_output);
}


/* Calculate the gradients for all neurons of a certain layer
*/
void MLP::computeGradients(int layer_number, double* expected_output)
{
    int i, j;

    if (layer_number == aNumberOfLayers - 1)
    {
        /* Alloc memory for the errors of the output layer */
        double* errors = (double*)calloc(aOutputSize, sizeof(double));

        /* Compute the errors for the output layer */
        for (int i = 0; i < aOutputSize; ++i)
        {
            errors[i] = expected_output[i] - aLayersOutputs[layer_number][i + 1];
        }
        /* Compute the derivative of the activation function for the output layer */
        sigmoidDerivative(aLayersOutputs[layer_number], aLayersDerivatives[layer_number], aOutputSize);

        for (int i = 0; i < aOutputSize; ++i)
        {
            aLocalGradients[layer_number][i] = errors[i] * aLayersDerivatives[layer_number][i];
        }
        free(errors);
    } else
    {
        sigmoidDerivative(aLayersOutputs[layer_number], aLayersDerivatives[layer_number], aLayersDim[layer_number]);
        double neuron_error;
        
        /* Compute the gradients between 2 consecutive layers */
        #pragma omp parallel for shared(i) private(neuron_error, j)
        for  (i = 0; i < aLayersDim[layer_number]; ++i)
        {
            neuron_error = 0;
            for (j = 0; j < aLayersDim[layer_number + 1]; ++j)
            {
                neuron_error += aLocalGradients[layer_number + 1][j] * aWeights[layer_number][i][j];
            }
            aLocalGradients[layer_number][i] = aLayersDerivatives[layer_number][i] * neuron_error;
        }
    }
}


/* Multiply a matrix of size 1xm with a matrix of size mxn, resulting
    a 1xn matrix.
*/
void MLP::matrixMultiply(double* mat1, double** mat2, double* result, int size_mat1, int size_mat2)
{
    int i, j;
    // int num_threads = omp_get_num_threads();
    
    #pragma omp parallel shared(result, mat1, mat2, i) private(j)
    {
        // num_threads = omp_get_num_threads();
        // std::cout << "#threads  "<< num_threads << std::endl;   
        #pragma omp for
        for (i = 0; i < size_mat2; ++i) {
            result[i] = 0.0;
            for (j = 0; j < size_mat1; ++j)
            {
                result[i] += (mat1[j] * mat2[j][i]);
            }
        }
    }
}


/* Function which computes the sigmoid activation for each element
    of an array.
*/
void MLP::sigmoidActivation(double* input, double* output, int length)
{
    /* Bias */
    output[0] = 1;

    for (int i = 0; i < length; ++i) 
    {
        /* Sigmoid function */
        output[i + 1] = 1.0 / (1.0 + exp(-input[i]));
    }
}


/* Function which computes the value of the sigmoid derivative function for each element
    of an array.
*/
void MLP::sigmoidDerivative(double* output, double* derivative, int length)
{
    for (int i = 0; i < length; ++i)
    {
        derivative[i] = output[i+1] * (1.0 - output[i+1]);
    }
}


void MLP::allocateWeights(double**** weights)
{
    /* Allocate memory for the array of weights matrices between layers */
    *weights = (double***)calloc(aNumberOfLayers - 1, sizeof(double**));

    for (int i = 0; i < aNumberOfLayers - 1; ++i)
    {
        /* Allocate memory for each weights matrix */
        (*weights)[i] = (double**)calloc(aLayersDim[i] + 1, sizeof(double*));
        for (int j = 0; j < aLayersDim[i] + 1; ++j)
        {
            (*weights)[i][j] = (double*)calloc(aLayersDim[i + 1], sizeof(double));
        }
    }
}


/* Destructor */
MLP::~MLP()
{    
    /* Free all the dynamically allocated memory */
    for (int i = 0; i < this->aNumberOfLayers; ++i)
    {
        free(aLayersInputs[i]);
        free(aLayersOutputs[i]);
        free(aLocalGradients[i]);
        free(aLayersDerivatives[i]);
    }
    free(aLayersInputs);
    free(aLayersOutputs);
    free(aLocalGradients);
    free(aLayersDerivatives);

    for (int i = 0; i < aNumberOfLayers - 1; ++i)
    {
        for (int j = 0; j < aLayersDim[i] + 1; ++j)
        {
            free(aWeights[i][j]);
            free(aCorrectWeights[i][j]);
        }
        free(aWeights[i]);
        free(aCorrectWeights[i]);
    }
    free(aWeights);
    free(aCorrectWeights);
}