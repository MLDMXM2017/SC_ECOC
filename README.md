SC-ECOC Code introduction
===
The directory structure
---
    /.
	/Data # Data fold
		detmatology.csv # Data example
	/SC_Code # Source code
		Classifier.py # Classifier definition
		Criterion.py # Criterion definition
		data_preprocess.py # Custom data preprocess
		Distance_Toolkit.py # Distance definition
		Feature_select.py # Feature selection method
		Matrix_Toolkit.py # Auxiliary methods for matrix generation
		SFFS.py # Forward floating search algorithm
	/sc_ecoc_result # results of SC-ECOC algorithm
	/main.py # Program entrance

Details of main.py
---
1. euclidean_distance(x, y): calculate the euclidean distance between x and y.
2. hamming_distance(x, y): calculate the hamming distance between x and y.
3. plot_data(classifier, test_label, predict_label, path): it plots coding matrix and confusion matrix for each classifier; the test_label and predict_label is used to plot confusion matrix; the path is the save path.
4. metric_fusion(root_path): it's used to calculate some result metrics.
5. runner(classifier, classifier_name, train_data, train_label, test_data, test_label, root_path, data_name, iteration): it's the main function in main.py file. it's use to runer this program and save experiment results. the classifier can be SC-ECOC or other algorithms. train_data, train_label, test_data and test_label should be given to train the classifier and test it. The iteration gives the round of test process. In the end the results are saved in sc_ecoc_result.

Details of Classifier.py
---
This file mainly contain the source code of SC-ECOC algorithm. It has following functions:
1. __init__(self, base_estimator=SVR, distance_measure=Distance_Toolkit.euclidean_distance,coverage='normal', fill=True, check=True, **estimator_params): it's used to initialize object. base_estimator defines the base learner for each matrix column. distance_measure defines the distance used in algorithm, which default is euclidean distance. coverage define the way to calculate coverage value, which can be 'normal' or 'distance'. fill decides wheather to fill 0 values in codematrix. check decides wheather to fine-tuning the codematrix. estimator_params can pass parameters to base_estimator.
2. create_matrix(self, data, label):it's used to create the codematrix. data and label are training data and training label.
3. fill_column_zero(self, data, label, column):it's used to fill 0 values in codematrix.
4. check(self, data, label): it's used to fine-tuning the codematrix.
5. check_matrix(self):it verifies the validity of the codematrix
6. predict(self, data):it's used to predict label for different samples.
7. fit(self, data, label):it's used to start train process based on data and label.

Development environment
---
- Pythoon 3.6
- Scipy
- Scikit-learn
- Numpy
- Pandas

The command to run demo
---
	python3.6 main.py
it uses ./Data/dermatology.csv dataset as training dataset

