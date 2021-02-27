#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include "utils.hpp"

using namespace std;



/* Function which reads the dataset from a csv file */ 
dataset_t read_csv_file(char* file_name)
{
	fstream input_file;
	string line;
	vector<vector<double> > elements;
	dataset_t dataset;

	char delim = ',';

	input_file.open(file_name);

	if(!input_file.is_open())
	{
		throw runtime_error("File is not open");
	}

	/* Ignore the first line of the csv file */
	getline(input_file, line);

	while (getline(input_file, line)) {
		stringstream line_stream(line);
		double val;
		vector<double> line_elems;

		/* Parse each line and add the elements into a vector*/
		while (getline(line_stream, line, delim)) {
			val = stof(line);
			line_elems.push_back(val);
	   	}

	   	/* Add line elements to the global vector */
	   	elements.push_back(line_elems);

	}

	/* Store the dataset in a double** matrix for memory efficiency reasons */
	dataset.aNbElements = elements.size();
	dataset.aFeatureSize = elements[0].size() - 1;
	dataset.aData = (double**)calloc(elements.size(), sizeof(double*));
	dataset.aLabels = (double*)calloc(elements.size(), sizeof(double));
	dataset.aOutputSize = 1;

	for (unsigned int i = 0; i < elements.size(); ++i)
	{
		dataset.aData[i] = (double*)calloc(dataset.aFeatureSize, sizeof(double));

		for (int j = 0; j < dataset.aFeatureSize; ++j)
		{
			dataset.aData[i][j] = elements[i][j];
		}
		dataset.aLabels[i] = elements[i][dataset.aFeatureSize];
	}
	return dataset;
}

/* Split the dataset in 2: train and test according to the ratio, which is a number
	in (0,1) interval. 
*/
void split_dataset(dataset_t dataset, dataset_t* train_set, dataset_t* test_set, double ratio)
{
	/* Initialize the test set structure */
	test_set->aNbElements = dataset.aNbElements * ratio;
	test_set->aFeatureSize = dataset.aFeatureSize;
	test_set->aData = (double**)calloc(test_set->aNbElements, sizeof(double*));
	test_set->aLabels = (double*)calloc(test_set->aNbElements, sizeof(double));
	test_set->aOutputSize = dataset.aOutputSize;

	/* Initialize the train set structure */
	train_set->aNbElements = dataset.aNbElements - test_set->aNbElements;
	train_set->aFeatureSize = dataset.aFeatureSize;
	train_set->aData = (double**)calloc(train_set->aNbElements, sizeof(double*));
	train_set->aLabels = (double*)calloc(train_set->aNbElements, sizeof(double));
	train_set->aOutputSize = dataset.aOutputSize;


	/* Copy the samples and labels from the dataset in the test set */
	for (int i = 0; i < test_set->aNbElements; ++i)
	{
		test_set->aData[i] = (double*)calloc(test_set->aFeatureSize, sizeof(double));

		for (int j = 0; j < test_set->aFeatureSize; ++j)
		{
			test_set->aData[i][j] = dataset.aData[train_set->aNbElements + i][j];
		}
		test_set->aLabels[i] = dataset.aLabels[train_set->aNbElements + i];
	}

	/* Copy the samples and labels from the dataset in the train set */
	for (int i = 0; i < train_set->aNbElements; ++i)
	{
		train_set->aData[i] = (double*)calloc(train_set->aFeatureSize, sizeof(double));

		for (int j = 0; j < train_set->aFeatureSize; ++j)
		{
			train_set->aData[i][j] = dataset.aData[i][j];
		}
		train_set->aLabels[i] = dataset.aLabels[i];
	}
}

/* Calculate the accuracy by dividing the number of correctly predicted 
	labels to the total number of labels
*/
double compute_accuracy(vector<double> predictions, double *true_labels)
{
	float total_correct = 0;

	for (size_t i = 0; i < predictions.size(); ++i)
	{
		if (predictions[i] == true_labels[i])
		{
			++total_correct;
		}
	}
	
	return total_correct / (float)predictions.size();
}
