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
    // int nb_hidden_layers = atoi(argv[3]);
    std::vector<int> architecture;

    // std::cout << "l_rate " <<learning_rate<< std::endl;
    // std::cout << "epochs " <<nb_epochs<< std::endl;
    // std::cout << "nb hidden layer " <<nb_hidden_layers<< std::endl;

    // std::cout << argc << std::endl;
    // std::string input_file = "../data/heart_disease_.csv";
	// std::string input_file = "../data/log2.csv";
    // std::cout << "the inpute file is: "<< argv[5] << std::endl;

    /* Read the dataset from disk */
    dataset_t dataset = read_csv_file(argv[4]);
    // std::cout << "Size dataset before split = "<<dataset.aNbElements << std::endl;
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
        // std::cout << token << std::endl;
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


    // std::cout << train_set.aNbElements << std::endl;
    // std::cout << test_set.aNbElements << std::endl;

    // for (int i = 0; i < train_set.aNbElements; ++i)
    // {
    //     std::cout << i << "_______";
    //     for (int j = 0; j < train_set.aFeatureSize; ++j)
    //     {
    //         std::cout << train_set.aData[i][j]<<" ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "________________________________________________________________________" << std::endl;
    // for (int i = 0; i < test_set.aNbElements; i++)
    // {
    //     std::cout << i << "__test_____";
    //     for (int j = 0; j < test_set.aFeatureSize; j++)
    //     {
    //         std::cout << test_set.aData[i][j]<<" ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << hidden_layers_str << std::endl;

    /* Instantiate the MLP class with 3 hidden layers */
    // int layers_size[] = {dataset.aFeatureSize, 5, 6, 7, dataset.aNbElements};

    // std::vector<int> architecture(layers_size, layers_size + sizeof(layers_size)/sizeof(int));

    /*************************** Training the NN ***************************/

    // int th_id;

    omp_set_num_threads(num_threads);
    // #pragma omp parallel private(th_id) shared (num_threads)
    // {
    //     th_id = omp_get_thread_num();
    //     // printf("Hello from %d\n ", th_id);
    //     #pragma omp barrier
    //     if (th_id == 0)
    //     {

    //         num_threads = omp_get_num_threads();
    //         std::cout << "From th_id="<<th_id<<" Total threads: "<<num_threads << std::endl;
    //     }
    // }

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

    // for (size_t i = 0; i < 10; i++)
    // {
    //     printf("%f\n", predicted_labels[i]);
    // }
    
    for (size_t i = 0; i < predicted_labels.size(); i++)
    {
        predicted_labels[i] = predicted_labels[i] > 0.5 ? 1 : 0;
    }
    // std::cout << "After discretization_________________-" << std::endl;
    // for (size_t i = 0; i < 10; i++)
    // {
    //     printf("%f\n", float(predicted_labels[i]));
    // }

    float acc = compute_accuracy(predicted_labels, test_set.aLabels);


    printf("accuracy = %f\n", acc);
    std::cout << "_________________________________________" << std::endl;
    return 0; 
}