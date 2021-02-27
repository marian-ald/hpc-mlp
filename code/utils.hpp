#include <errno.h>


#define DIE(assertion, call_description)    \
	do {								    \
		if (assertion) {					\
			fprintf(stderr, "(%s, %d): ",	\
					__FILE__, __LINE__);	\
			perror(call_description);		\
			exit(errno);					\
		}							        \
	} while (0)

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

// void cont_to_discrete(std::vector<double> &pred);