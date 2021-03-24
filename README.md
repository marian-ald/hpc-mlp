# Parallelizing the Multilayer perceptron in C++ using openmp library

For testing, go in the __hpc-mlp/code__ directory and compile it by running:

    make

To execute the program, run: 
    
    ./mlp learning_rate nb_epochs l1,l2,etc. input_file.csv nb_threads

For example:

    ./mlp 0.001 100 50,60,50,10 ../data/heart_disease_norm.csv 2


The full documentation of the project can be found in `report.pdf`.
