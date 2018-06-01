#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
#include <vector>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

bool buildClassifier(string &dataFile, string &saveFile, string &loadFile);
bool readNumClassData(string &filename, int varCount, Mat *data, Mat *responses);
template<typename T>
Ptr<T> loadClassifier(const string &loadFile);
void testAndSaveClassifier(const Ptr<StatModel> &model, const Mat &data, const Mat &responses, 
							int nTrainSamples, int rDelta, const string &saveFile);