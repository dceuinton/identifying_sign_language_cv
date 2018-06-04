
#include <stdio.h>
#include <string>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>
#include "opencv2/ml/ml.hpp"

#include "filters.h"
#include "classifier.h"

// #define GENERATEDESCRIPTORS
// #define TESTSAMPLES
// #define RUNONGESTURES
#define REGULAR
// #define MY_CLASSIFIER_STUFF
// #define TEST_MY_CLASSIFIER

using namespace std;
using namespace cv;
using namespace cv::ml;

const char *imageDescriptorsFile = "descriptors.data";
const char *imageNamesFile = "images.txt";
const char *ellipticalFourierDescriptorFile = "Descriptors.txt";
const char *testImageNamesFile = "testImages.txt";
const char *testImageDescriptorsFile = "testDescriptors.txt";
const char *classifier = "a3classifier.xml";
const char *classOrderFile = "classOrderFile.txt";
const char *slimShady = "theRealSlimShadyClassifier.xml";

void saveOrderOfClassesToFile(const char *classOrderFile);
void ellipticFourierDescriptors(vector<Point> &contour, vector<float> &CE);
void writeDescriptors(const char *filename, vector<float> &CE);
void writeClass(const char *filename, char identifier);
void clearContentsOfFile(const char *filename);
void readInImageNames(const char *imageNamesFile);//, vector<vector<String>> &imageNames);
void printImageNames();
vector<float> generateEllipticalFourierDescriptors(const char *filename);
void writeDescriptors(const char *imageFileName, const char *outputFilename);
template<typename T>
static Ptr<T> load_classifier(const string &filename);
void initKeyForClasses();
float getPrediction(const char *filename);
string getValue(float prediction);
bool isInString(char element, string word);
void sortFileIntoOrder(const char *filename);
template<typename T>
void printVector(vector<T> &vec);
void putLabel(cv::Mat &mat, const std::string label, cv::Point &p);

// This vector should have the order that the images are stored in. So the the letters index is the same as the 
// index of the vector<string>'s index in vector<vector<String>>
vector<char> classOrder;
vector<string> imageNames;
map<char, int> keyForClasses;
Ptr<ANN_MLP> model;

int correctPredictions = 0;
int total = 0;

int main(int argc, char const *argv[]) {

	initKeyForClasses();
	const char *filename;	
	// model = load_classifier<ANN_MLP>(classifier);
	model = load_classifier<ANN_MLP>(slimShady);

	if (argc > 1) {
		filename = argv[1];	
	} else {
		cout << "Starting Camera" << endl;
		filename = "./gestures/test.png";

		// Camera function
	}

#ifdef MY_CLASSIFIER_STUFF

	Mat data, responses;

	string descriptorFile = "descriptors.data";
	string saveFile = "theRealSlimShadyClassifier.xml";
	string empty = "";

	buildClassifier(descriptorFile, saveFile, empty);

#endif

#ifdef TEST_MY_CLASSIFIER

	Mat data, responses;

	string descriptorFile = "descriptors.data";
	string loadFile = "theRealSlimShadyClassifier.xml";
	string empty = "";

	buildClassifier(descriptorFile, empty, loadFile);

#endif

	// --------------------------------------------------------------------------------

#ifdef GENERATEDESCRIPTORS 
	printf("Generating descriptors:\n");

	const char *input = imageNamesFile;
	const char *output = imageDescriptorsFile;

	writeDescriptors(input, output);

	// printf("%i, out of %i\n", correctPredictions, total);

	sortFileIntoOrder(output);

#endif

	// --------------------------------------------------------------------------------

#ifdef TESTSAMPLES	
	printf("Testing Samples:\n");

	float r;

	// 1
	// Mat sample0 = (Mat_<float>(1, 9) << 0.483537,0.0897202,0.0896418,0.0904506,0.0587337,0.027807,0.00611194,0.0210563,0.0212424);
	Mat sample0 = (Mat_<float>(1, 9) << 0.192112,0.209336,0.0675378,0.0860401,0.0592248,0.0402964,0.0492288,0.0460786,0.0377082);
    r = model->predict(sample0);
    printf("Predicted: %f\n", r);

    // 2 
    Mat sample1 = (Mat_<float>(1, 9) << 0.288093,0.263243,0.150342,0.13571,0.0671558,0.0666056,0.0320719,0.0334513,0.0270351);
    r = model->predict(sample1);
    printf("Predicted: %f\n", r);

    // 3
    Mat sample2 = (Mat_<float>(1, 9) << 0.863456,0.546805,0.379552,0.186811,0.167384,0.0793804,0.0974595,0.0781477,0.0230621);
    r = model->predict(sample2);
    printf("Predicted: %f\n", r);

    // 4
    Mat sample3 = (Mat_<float>(1, 9) << 0.67012,0.196405,0.460634,0.20463,0.116233,0.0743352,0.0783094,0.0690908,0.0195137);
    r = model->predict(sample3);
    printf("Predicted: %f\n", r);

    // 5
    Mat sample4 = (Mat_<float>(1, 9) << 1.20807,0.374261,0.27939,0.320042,0.273963,0.15068,0.119287,0.129897,0.061082);
    r = model->predict(sample4);
    printf("Predicted: %f\n", r);

    // 6
    Mat sample5 = (Mat_<float>(1, 9) << 0.583613,0.341947,0.123211,0.163423,0.101956,0.184059,0.230072,0.186048,0.0613964);
    r = model->predict(sample5);
    printf("Predicted: %f\n", r);

	// 7
	// Mat sample6 = (Mat_<float>(1, 9) << 0.997807,0.13736,0.304255,0.247654,0.223244,0.141323,0.107183,0.0630595,0.0491406);
	// r = model->predict(sample6);
	// printf("Predicted: %f\n", r);

	// // 8
	// Mat sample7 = (Mat_<float>(1, 9) << 0.778391,0.506046,0.385205,0.296748,0.228848,0.141737,0.169707,0.0558902,0.11081);
	// r = model->predict(sample7);
	// printf("Predicted: %f\n", r);

	// // 9
	// Mat sample8 = (Mat_<float>(1, 9) << 0.522996,0.159145,0.177614,0.302856,0.269031,0.180462,0.0505989,0.131701,0.0813645);
	// r = model->predict(sample8);
	// printf("Predicted: %f\n", r);

	// // 10
	// // Mat sample9 = (Mat_<float>(1, 9) << 0.750121,0.13988,0.22171,0.238601,0.203625,0.113773,0.161233,0.0718545,0.0621363);
	// Mat sample9 = (Mat_<float>(1, 9) << 0.326396,0.24913,0.19725,0.112993,0.085271,0.0901283,0.0600791,0.0863757,0.0327872);
	// r = model->predict(sample9);
	// printf("Predicted: %f\n", r);

#endif

	// --------------------------------------------------------------------------------

#ifdef RUNONGESTURES

	// printf("%s\n", filename);
	float prediction = getPrediction(filename);
	string value = getValue(prediction);
	if (isInString(filename[17], value)) { 
		return 1; 
	} else { 
		// vector<float> CE = generateEllipticalFourierDescriptors(filename);
		// printf("%s, %c, %f, %s", filename, filename[17], prediction, value.c_str());
		// printf(" %f,%f,%f,%f,%f,%f,%f,%f,%f\n", CE[1], CE[2], CE[3], CE[4], CE[5], CE[6], CE[7], CE[8], CE[9]);
		return 0; 
	}

#endif

#ifdef REGULAR

	float prediction = getPrediction(filename);
	string value = getValue(prediction);

	printf("Value: %i\n", (int) prediction);

	Mat image = imread(filename);

	Point valueLoc; valueLoc.x = 10; valueLoc.y = 50;

	// Write the thing on it
	// putText(image, value, valueLoc, FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 0), 2.0);
	putLabel(image, value, valueLoc);

	imshow(filename, image);
	waitKey(0);

#endif
	
	return 0;
}

// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------

vector<float> generateEllipticalFourierDescriptors(const char *filename) {
	Mat *src = new Mat();
	*src = imread(filename);
	Mat copy(src->size(), src->type(), Scalar(0,0,0));
	Mat *binaryImage = threshold(src, 2, THRESHOLD_BINARY, false);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(*binaryImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	int idx = 0;
	for (; idx >=0; idx = hierarchy[idx][0]) {
		Scalar colour(rand() & 255, rand() & 255, rand() & 255);
		drawContours(copy, contours, idx, colour, 1, 8, hierarchy);
	}

	// imshow(filename, copy);
	// waitKey(0);

	vector<float> CE;
	ellipticFourierDescriptors(contours[0], CE);

	delete src;
	src = NULL;
	return CE;
}

void startCamera() {
	cv::VideoCapture stream1(0);

	if (!stream1.isOpened()) {
		printf("ERROR: Camera isn't opening.\n");
		return;
	}

	string windowName = "Camera Feed";
	int numberOfFrames = 120, frames = 0;
	double fps;
	time_t tStart, tFinish;
	Mat cameraFrame;

	time(&tStart);

	while(true) {
		stream1.read(cameraFrame);
		frames++;

		time(&tFinish);
		double seconds = difftime(tFinish, tStart);
		fps = frames/seconds;
		Point predictionPoint;
		Point fpsPoint; fpsPoint.x = 50; fpsPoint.y = 200;

		putLabel(cameraFrame, to_string(fps).substr(0,3), fpsPoint);
	}
}

void writeDescriptors(const char *imageFileName, const char *outputFilename) {
	readInImageNames(imageFileName);
	clearContentsOfFile(outputFilename);

	for (int i = 0; i < imageNames.size(); i++) {
		vector<float> CE = generateEllipticalFourierDescriptors(imageNames[i].c_str());
		char identifier = imageNames[i][17];
		float correctClass = keyForClasses[identifier];

		// Mat sample = (Mat_<float>(1, 9) << CE[1],CE[2],CE[3],CE[4],CE[5],CE[6],CE[7],CE[8],CE[9]);
		// float r = model->predict(sample);
		// printf("Predicted: %f, correct: %f\n", r, correctClass);

		// if (r == correctClass) {
		// 	correctPredictions++;
		// }
		// total++;

		// printf("%s,  %c, %i\n", imageNames[i].c_str(), identifier, keyForClasses[identifier]);
		// cout << "identifier " << identifier << endl;
		writeClass(outputFilename, identifier);
		writeDescriptors(outputFilename, CE);
	}
}

template<typename T>
void printVector(vector<T> &vec) {
	cout << "Printing vector: " << endl;
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i] << ",";
	}
	cout << endl;
}

void initKeyForClasses() {
	keyForClasses['0'] = 1;
	keyForClasses['1'] = 2;
	keyForClasses['2'] = 3;
	keyForClasses['3'] = 4;
	keyForClasses['4'] = 5;
	keyForClasses['5'] = 6;
	keyForClasses['6'] = 7;
	keyForClasses['7'] = 8;
	keyForClasses['8'] = 9;
	keyForClasses['9'] = 10;
	keyForClasses['a'] = 11;
	keyForClasses['b'] = 12;
	keyForClasses['c'] = 13;
	keyForClasses['d'] = 2;
	keyForClasses['e'] = 11;
	keyForClasses['f'] = 14;
	keyForClasses['g'] = 2;
	keyForClasses['h'] = 15;
	keyForClasses['i'] = 16;
	keyForClasses['j'] = 16;
	keyForClasses['k'] = 3;
	keyForClasses['l'] = 17;
	keyForClasses['m'] = 11;
	keyForClasses['n'] = 11;
	keyForClasses['o'] = 1;
	keyForClasses['p'] = 18;
	keyForClasses['q'] = 19;
	keyForClasses['r'] = 2;
	keyForClasses['s'] = 11;
	keyForClasses['t'] = 11;
	keyForClasses['u'] = 15;
	keyForClasses['v'] = 3;
	keyForClasses['w'] = 7;
	keyForClasses['x'] = 20;
	keyForClasses['y'] = 21;
	keyForClasses['z'] = 2;
}

float getPrediction(const char *filename) {
	vector<float> CE = generateEllipticalFourierDescriptors(filename);
	// printVector(CE);
	Mat sample = (Mat_<float>(1, 19) << CE[1], CE[2], CE[3], CE[4], CE[5], CE[6], CE[7], CE[8], CE[9],
										CE[10], CE[11], CE[12], CE[13], CE[14], CE[15], CE[16], CE[17], 
										CE[18], CE[19]);
	float r = model->predict(sample);
	return r;
}

bool isInString(char element, string word) {
	if (find(word.begin(), word.end(), element) != word.end()) {
		return true;
	} else {
		return false;
	}
}

string getValue(float prediction) {
	string result = "Undefined";
	if (prediction == 1.0f) {
		result = "[0, o]";
	}
	else if (prediction == 2.0f) {
		result = "[1, d, g, r, z]";
	}
	else if (prediction == 3.0f) {
		result = "[2, k, v]";
	} else if (prediction == 4.0f) {
		result = "[3]";
	} else if (prediction == 5.0f) {
		result = "[4]";
	} else if (prediction == 6.0f) {
		result = "[5]";
	} else if (prediction == 7.0f) {
		result = "[6, w]";
	} else if (prediction == 8.0f) {
		result = "[7]";
	} else if (prediction == 9.0f) {
		result = "[8]";
	} else if (prediction == 10.0f) {
		result = "[9]";
	} else if (prediction == 11.0f) {
		result = "[a, e, m, n, s, t]";
	} else if (prediction == 12.0f) {
		result = "[b]";
	} else if (prediction == 13.0f) {
		result = "[c]";
	} else if (prediction == 14.0f) {
		result = "[f]";
	} else if (prediction == 15.0f) {
		result = "[h, u]";
	} else if (prediction == 16.0f) {
		result = "[i, j]";
	} else if (prediction == 17.0f) {
		result = "[l]";
	} else if (prediction == 18.0f) {
		result = "[p]";
	} else if (prediction == 19.0f) {
		result = "[q]";
	} else if (prediction == 20.0f) {
		result = "[x]";
	} else if (prediction == 0.0f) {
		result = "[y]";
	} 
	// else if (prediction == 22.0f) {
	// 	result = "[r]";
	// } else if (prediction == 23.0f) {
	// 	result = "[x]";
	// } else if (prediction == 24.0f) {
	// 	result = "[y]";
	// }

	return result;
}

// The C implementation that I borrowed from the lecture slides and notes. 
void ellipticFourierDescriptors(vector<Point> &contour, vector<float> &CE) {
	vector<float> ax, ay, bx, by;
	int m = contour.size();
	// int n = 15;                        // Number of CEs 
	int n = 20;                        // Number of CEs 
	float t = (2*M_PI)/m;

	for (int k = 0; k < n; k++) {
		ax.push_back(0.0);    ay.push_back(0.0);
		bx.push_back(0.0);    by.push_back(0.0);

		for (int i = 0; i < m; i++) {
			ax[k] = ax[k] + contour[i].x * cos((k + 1) * i * t);
			bx[k] = bx[k] + contour[i].x * sin((k + 1) * i * t);
			ay[k] = ay[k] + contour[i].y * cos((k + 1) * i * t);			
			by[k] = by[k] + contour[i].y * sin((k + 1) * i * t);
		}
		ax[k] = ax[k]/m;
		ay[k] = ay[k]/m;
		bx[k] = bx[k]/m;
		by[k] = by[k]/m;
	}

	for (int k = 0; k < n; k++) {
		CE.push_back(sqrt((ax[k] * ax[k] + ay[k] * ay[k])/(ax[0] * ax[0] + ay[0] *ay[0])) + 
					 sqrt((bx[k] * bx[k] + by[k] * by[k])/(bx[0] * bx[0] + by[0] *by[0])));
	}
}

void saveOrderOfClassesToFile(const char *classOrderFile) {
	clearContentsOfFile(classOrderFile);
	ofstream file(classOrderFile, ofstream::out | ofstream::app);
	for (int i = 0; i < classOrder.size(); i++) {
		file << classOrder[i] << endl;
	}

	file.close();
}

void clearContentsOfFile(const char *filename) {
	ofstream file(filename, ofstream::out | ofstream::trunc);
	file.close();
}

void writeDescriptors(const char *filename, vector<float> &CE) {
	ofstream file(filename, ofstream::out | ofstream::app);

	for (int i = 1; i < CE.size(); i++) {
		file << "," << CE[i];
	}
	file << endl;

	file.close();
}

void writeClass(const char *filename, char identifier) {
	ofstream file(filename, ofstream::out | ofstream::app);

	file << (keyForClasses.at(identifier));

	file.close();
}

bool contains(vector<char> &order, char element) {
	if (find(order.begin(), order.end(), element) != order.end()) {
		return true;
	} else {
		return false;
	}
}

int getIndex(vector<char> &order, char element) {
	if (contains(order, element)) {
		return find(order.begin(), order.end(), element) - order.begin();	
	} else {
		return -1;
	}	
}

void printImageNames() {
	printf("Printing Image Names:\n");
	for (int i = 0; i < imageNames.size(); i++) {
			printf("%s\n", imageNames[i].c_str());
	}
}

void readInImageNames(const char *imageNamesFile) {//, vector<vector<String>> &imageNames) {
	// Read file into buffer
	ifstream input(imageNamesFile);
	stringstream ss;
	ss << input.rdbuf();
	input.close();

	string word;

	while (getline(ss, word)) {
		imageNames.push_back(word);
	}
}

// Will write the class and fourier descriptors from the file of image names into the file outputFileName

template<typename T> 
static Ptr<T> load_classifier(const string &filename) {

	Ptr<T> model = StatModel::load<T>( filename );
    if( model.empty() )
        cout << "Could not read the classifier " << filename<< endl;
    else
        cout << "The classifier " << filename << " is loaded.\n";

    return model;
}

bool mySortFunction(const string &lhs, const string &rhs) {
	stringstream lhsSS;
	stringstream rhsSS;
	lhsSS.str(lhs);
	rhsSS.str(rhs);

	string lNum;
	string rNum;

	getline(lhsSS, lNum, ',');
	getline(rhsSS, rNum, ',');

	int lInt = atoi(lNum.c_str());
	int rInt = atoi(rNum.c_str());

	printf("Strings %s, %s\n", lNum.c_str(), rNum.c_str());
	printf("Numbers %i, %i\n", lInt, rInt);


	// printf("%c, %c\n", lhs[0], rhs[0]);
	if (lInt < rInt) {
		return true;
	} else {
		return false;
	}
}

void sortFileIntoOrder(const char *filename) {
	ifstream file(filename);
	stringstream ss;
	ss << file.rdbuf();
	file.close();

	string line;

	vector<string> lines;

	while(getline(ss, line)) {
		// stringstream ss2;
		// ss2.str(line);
		// string firstNum;
		// getline(ss2, firstNum, ',');
		// printf("%s\n", firstNum.c_str());
		// printf("%s\n", line.c_str());
		lines.push_back(line);
	}

	printf("Step 1\n");

	sort(lines.begin(), lines.end(), mySortFunction);

	printf("Step 2\n");

	clearContentsOfFile(filename);

	ofstream output(filename, ofstream::out | ofstream::app);

	printf("Step 3\n");

	for (int i = 0; i < lines.size(); i++) {
		output << lines[i] << endl;
	}

	output.close();
}

// void testDescriptors(const char *dataFile) {
// 	ifstream file(dataFile, ifstream::in);
// }

void putLabel(cv::Mat &mat, const std::string label, cv::Point &p) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.8;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(mat, p+cv::Point(0, baseline), p + cv::Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
    cv::putText(mat, label, p, fontface, scale, CV_RGB(255,255,255), thickness, 8);
}
