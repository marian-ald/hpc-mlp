#include <omp.h>

#include <string>
#include <chrono>

#include "MLP.hpp" 
#include "utils.hpp"


using namespace std;
using namespace std::chrono;

int main(int argc, char** argv)
{ 
    /* Check if the program is called with the right parameters */
    if (argc != 6)
    {
        std::cout << "Invalid number of arguments. Expected 3. The correct command is:" << std::endl;
        std::cout << "  ./mlp learning_rate nb_epochs nb_hidden_layers l1,l2,etc. input_file.csv nb_threads" << std::endl;
        std::cout << "Don't add any spaces in the <<l1,l2,...>> enumeration" << std::endl;
        exit(0);
    }

    double learning_rate = atof(argv[1]);
    int nb_epochs = atoi(argv[2]);
    int num_threads = atoi(argv[5]);
    std::vector<int> architecture;


    /* Read the dataset from disk */
    dataset_t dataset = read_csv_file(argv[4]);
    dataset_t train_set, test_set;
    split_dataset(dataset, &train_set, &test_set, 0.2);
    vector<double> predicted_labels;


    /* The input layer of the NN is equal to the feature vector size*/
    architecture.push_back(dataset.aFeatureSize);

    string delimiter = ",";
    string token, hidden_layers_str(argv[3]);
    size_t pos = 0;

    /* Parse the hidden layers dimensions string and append them as integers to the 
        architecture vector */
    while ((pos = hidden_layers_str.find(delimiter)) != std::string::npos) {
        token = hidden_layers_str.substr(0, pos);
        architecture.push_back(stoi(token));
        hidden_layers_str.erase(0, pos + delimiter.length());
    }
    architecture.push_back(stoi(hidden_layers_str));
    architecture.push_back(dataset.aOutputSize);

    std::cout << "#threads = "<< num_threads << std::endl;
    std::cout << "#epochs = "<< nb_epochs << std::endl;
    std::cout << "lr = "<< learning_rate << std::endl;
    std::cout << "#layers = "<< architecture.size() << std::endl;
    
    std::cout << "Architecture: " << endl;
    for (size_t i = 0; i < architecture.size(); i++)
    {
        std::cout << "layer" << i << " = " << architecture[i] << endl;
    }


    omp_set_num_threads(num_threads);

    /* Create an instance of the MLP with the desired hyperparameters: #layers, #neurons/layer,
        #epochs, learning_rate */
    MLP neuralNetwork(architecture, nb_epochs, learning_rate);

    auto start = high_resolution_clock::now();

    /* Train the neural network on the dataset */
    neuralNetwork.train(train_set);

    /* The predicted labels are written in 'aLabels' fields of the test dataset structure */
    predicted_labels = neuralNetwork.test(&test_set);

    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 

    cout << "Time taken by function: "<< duration.count() / float(1000000)<< " s" << endl;
    
    for (size_t i = 0; i < predicted_labels.size(); i++)
    {
        predicted_labels[i] = predicted_labels[i] > 0.5 ? 1 : 0;
    }

    float acc = compute_accuracy(predicted_labels, test_set.aLabels);

    printf("Accuracy = %f\n", acc);

    return 0; 
}