// Open CV
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

const Scalar WHITE_COLOR = Scalar(255,255,255);
const int testStep = 5;

//Standard
#include <stdio.h>
#include <iostream>
#include <map>
#include <fstream>
#include <stdlib.h>

#include "CsvParser.h"

using namespace std;

typedef vector< vector<string> > MatrixStr;


void PrepareDataSet(CsvParser &myRawData, Mat &dataSet, vector<int>selColIndices, int type=CV_32FC1)
{
	dataSet = Mat(myRawData.getRows(),selColIndices.size(),type);
	
	for(int i = 0 ; i<myRawData.getRows() ; i++)
	{		
		for(int j = 0 ; j<selColIndices.size() ; j++)
			dataSet.at<float>(i,j) = (int)myRawData.stringToNum(myRawData.matrix[i][selColIndices[j]]);
	}
}

//
//void CreateDataSet(CsvParser &myRawData, Mat &trainSet, Mat &labelSet, int type)
//{
//	trainSet = Mat(myRawData.getRows(),myRawData.getCols()-1,type);
//	labelSet = Mat(myRawData.getRows(),1,type);
//	for(int i = 0 ; i<myRawData.getRows() ; i++)
//	{
//		labelSet.at<float>(i,0) = (int)myRawData.stringToNum(myRawData.matrix[i][myRawData.getCols()-1]);
//		for(int j = 0 ; j<myRawData.getCols()-1 ; j++)
//			trainSet.at<float>(i,j) = (int)myRawData.stringToNum(myRawData.matrix[i][j]);
//	}
//}

void trainingAccuracyReport(Mat &dataSet, Mat &labelSet , CvRTrees &rtrees) 
{
	cout << "Calculating Predictions ...\n";
	unsigned int truePositive = 0;
	unsigned int trueNegative = 0;
	unsigned int falsePositive = 0;
	unsigned int falseNegative = 0;


	Mat testSample(1, dataSet.cols, CV_32FC1 );
	for( int row = 0; row < dataSet.rows; row++)
	{
		for( int col = 0; col < dataSet.cols; col++)
			testSample.at<float>(0,col) = dataSet.at<float>(row,col);
		int predictedLabel = (int)rtrees.predict(testSample);
		int trueLabel = (int)labelSet.at<float>(row,0);
		if ( predictedLabel == 1 && trueLabel == 1 ) ++truePositive;
		else if ( predictedLabel == 1 && trueLabel == 0 ) ++falsePositive;
		else if ( predictedLabel == 0 && trueLabel == 1) ++falseNegative;
		else if ( predictedLabel == 0 && trueLabel == 0) ++trueNegative;
		else cout << "Invalid Classification (make sure you are using binary labels)" << endl; //invalid
	}

	double accuracy = (truePositive + trueNegative) * 100.0 / (truePositive + falsePositive + trueNegative + falseNegative);
	double precision = truePositive * 100.0 / (truePositive + falsePositive);
	double true_negative_rate = trueNegative * 100.0 / (trueNegative + falsePositive);
	double recall = truePositive * 100.0 / (truePositive + falseNegative);

	cout
		<< "-----------------------------------------------------------------------\n"
		<< "Training Set Classification:\n"
		<< "-----------------------------------------------------------------------\n"
		<< "True Positive     : " << truePositive << "\n"
		<< "False Positive    : " << falsePositive << "\n"
		<< "True Negative     : " << trueNegative << "\n"
		<< "False Negative    : " << falseNegative << "\n"
		<< "Accuracy          : " << accuracy << "%\n"
		<< "Precision         : " << precision << "%\n"
		<< "True negative rate: " << true_negative_rate << "%\n"
		<< "Recall            : " << recall << "%\n"
		<< "-----------------------------------------------------------------------\n";
}

void runRandomForest(Mat &dataSet, Mat &labelSet, CvRTrees  &rtrees)
{
	float priors[] = {1,1}; // same bias for all class labels
	CvRTParams  params( 25, // max_depth,
		5, // min_sample_count,
		0.f, // regression_accuracy,
		false, // use_surrogates,
		16, // max_categories,
		0, // priors,
		false, // calc_var_importance,
		3, // nactive_vars,
		500, // max_num_of_trees_in_the_forest,
		0, // forest_accuracy,
		CV_TERMCRIT_ITER // termcrit_type
		);

	rtrees.train( dataSet, CV_ROW_SAMPLE, labelSet, Mat(), Mat(), Mat(), Mat(), params );

}

void SelectFeatures(CsvParser &myParser, vector<int>selColIndices )
{
	myParser.selectCol(selColIndices);
	//Sex
	myParser.replace(1,"female","0");
	myParser.replace(1,"male","1");
	////Age
	int countAge = 0;
	double sumAge = 0;
	for(int i = 0 ; i<myParser.getRows() ; i++)
		if (myParser.matrix[i][2] != "")
		{
			++countAge;
			sumAge += myParser.stringToNum(myParser.matrix[i][2]);
		}
		double mean = sumAge/(double)countAge;
		for(int i = 0 ; i<myParser.getRows() ; i++)
			if (myParser.matrix[i][2] == "")
				myParser.matrix[i][2] = myParser.numToString(mean);
		//Embarked
		myParser.replace(5,"Q","0");
		myParser.replace(5,"S","1");
		myParser.replace(5,"C","2");
}


int main()
{
//	freopen("output.out", "w", stdout);
	//Training Parsing Phase
	CsvParser myTrainCSV;
	myTrainCSV.readMatrix("D:\\Osama\\Programming\\Projects\\Titanic - Machine Learning from Disaster\\Data\\train.csv");
	const int trainSelColIndices[] = {2,4,5,6,7,11,1};
	const int trainColSize = 7;
	SelectFeatures(myTrainCSV, vector<int>(trainSelColIndices , trainSelColIndices+trainColSize));
	//PassengerId - Survived - Pclass - Name - Sex - Age - SibSp - Parch - Ticket - Fare - Cabin - Embarked
	//Pclass - Sex - Age - SibSp - Parch - Embarked - Survived
	// 0		1	  2		 3		 4		 5		     6			

	Mat TrainData;
	const int trainFeature[] = {0,1,2,3,4,5};
	const int trainFSize = 6;
	PrepareDataSet(myTrainCSV,TrainData,vector<int>(trainFeature , trainFeature+trainFSize));
	Mat LabelData;
	const int classCol[] = {6};
	PrepareDataSet(myTrainCSV,LabelData,vector<int>(classCol , classCol+1));

	// Build Forest on Train Data
	CvRTrees  randomForest;
	runRandomForest(TrainData, LabelData, randomForest);
	trainingAccuracyReport(TrainData, LabelData, randomForest);


	//Make Test Predictions
	CsvParser myTestCSV;
	myTestCSV.readMatrix("D:\\Osama\\Programming\\Projects\\Titanic - Machine Learning from Disaster\\Data\\test-clean.csv");
	const int testSelColIndices[] = {1,3,4,5,6,10};
	const int testColSize = 6;
	SelectFeatures(myTestCSV, vector<int>(testSelColIndices , testSelColIndices+testColSize));

	Mat TestData;
	const int testFet[] = {0,1,2,3,4,5};
	const int testFSize = 6;
	PrepareDataSet(myTestCSV,TestData,vector<int>(testFet , testFet+testFSize));

	vector<int> predictedLabels;
	Mat testSample(1, TestData.cols, CV_32FC1 );
		for( int row = 0; row < TestData.rows; row++)
		{
			for( int col = 0; col < TestData.cols; col++)
				testSample.at<float>(0,col) = TestData.at<float>(row,col);
			int predictedLabel = (int)randomForest.predict(testSample);
			predictedLabels.push_back(predictedLabel);
		}

	// Output Prediction File
	CsvParser outputCSV;
	outputCSV.matrix.resize(TestData.rows+1);
	outputCSV.matrix[0].resize(2);
	outputCSV.matrix[0][0]="PassengerId",outputCSV.matrix[0][1]="Survived";
	for(int i=0; i<TestData.rows ; i++)
	{
		outputCSV.matrix[i+1].resize(2);
		outputCSV.matrix[i+1][0] = outputCSV.numToString(892+i);
		outputCSV.matrix[i+1][1] = outputCSV.numToString(predictedLabels[i]);
	}
	outputCSV.writeMatrix();

	system ("PAUSE");
	return 0;
}

