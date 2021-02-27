
#ifndef DATASET
#define DATASET

typedef struct dataset
{	
	int aNbElements;
	int aFeatureSize;
	int aOutputSize;
	double** aData;
	double* aLabels;
} dataset_t;

#endif

dataset_t read_csv_file(char* file_name);

void split_dataset(dataset_t dataset, dataset_t* train_dataset, dataset_t* test_set, double ratio);

double compute_accuracy(std::vector<double> predictions, double *true_labels);
