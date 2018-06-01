#include "classifier.h"

void printMatSize(const char *name, Mat &mat) {
	cout << name << "'s size: " << "[" << mat.rows << ", " << mat.cols << "]" << endl;
}

template<typename T>
void printMat(const char *name, Mat &mat, T t) {
	T *row;
	cout << name << endl;
	for (int i = 0; i < mat.rows; i++)	{
		row = mat.ptr<T>(i);
		for (int j = 0; j < mat.cols; j++) {
			cout << row[j] << ", ";
		}
		cout << endl;
	}
}

inline TermCriteria TC(int iterations, double eps) {
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iterations, eps);
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

bool buildClassifier(string &dataFile, string &saveFile, string &loadFile) {
	const int classCount = 24;
	Mat data;
	Mat responses;

	bool ok = readNumClassData(dataFile, 9, &data, &responses);
	if (!ok) {
		return ok;
	}

	Ptr<ANN_MLP> model;

	int nSamplesAll = data.rows;
	int nTrainSamples = (int) (nSamplesAll * 1.0); // Split with the 1.0 param

	if (!loadFile.empty()) {
		model = loadClassifier<ANN_MLP>(loadFile);
		if (model.empty()) {
			return false;
		}
	} else {
		Mat trainData = data.rowRange(0, nTrainSamples);
		Mat trainResponses = Mat::zeros(nTrainSamples, classCount, CV_32F);

		// 1. Unroll the responses
		cout << "Unrolling the responses... " << endl;
		for (int i = 0; i < nTrainSamples; i++) {
			int classLabel = responses.at<int>(i); // cout << "Make sure this isn't above 24 " << responses.at<int>(i) << endl; // This is all good

			cout << "Labels " << classLabel << endl;
			trainResponses.at<float>(i, classLabel) = 1.f;
		}
		// printMat("trainResponses", trainResponses, 0.0f);

		// 2. Train the classifier
		int layerSize[] = {data.cols, 100, 100, classCount};
		cout << "Sizeof layerSize " << sizeof(layerSize) << " sizeof layerSize[0] " << sizeof(layerSize[0]) << endl;
		int nLayers = (int)(sizeof(layerSize)/sizeof(layerSize[0]));
		cout << "Layers: " << nLayers << endl;
		Mat layerSizes(1, nLayers, CV_32S, layerSize);

		int method = ANN_MLP::BACKPROP;
		double methodParameter = 0.003;
		int maxIterations = 300;

		Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainResponses);

		cout << "Training classifier. This may take a few minutes...\n";
		model = ANN_MLP::create();
		model->setLayerSizes(layerSizes);
		model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
		model->setTermCriteria(TC(maxIterations, 0));
		model->setTrainMethod(method, methodParameter);
		model->train(tData);
		cout << endl;
	}

	testAndSaveClassifier(model, data, responses, nTrainSamples, 0, saveFile);

	return true;
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

int getClass(const char *buffer, char delim) {
	int index = 0;;

	while (*buffer != delim) {
		index ++;
		buffer++;
	}

	// cout << "The index: " << index << endl;

	char num[index];
	for (int i = 0; i < index; i ++) {
		num[i] = buffer[i - index];
	}

	// cout << "Num: " << num << endl;

	string number(num);
	int output = atoi(number.c_str());

	// cout << "Output " << output << endl;

	return output;
}

int getBytesToNumber(const char *buffer, char delim) {
	int bytes = 0;
	while (*buffer != delim) {
		bytes ++;
		buffer++;
	}
	return bytes;
}

bool readNumClassData(string &filename, int varCount, Mat *_data, Mat *_responses) {

	const int M = 1024;
	char buf[M+2];

	Mat elPtr(1, varCount, CV_32F);

	// printMatSize("elPtr", elPtr);

	int i;
	vector<int> responses;

	_data->release();
	_responses->release();

	FILE *f = fopen(filename.c_str(), "rt");
	if (!f) {
		cout << "Couldn't read file: " << filename << endl;
		return false;
	}

	for (;;) {
		char *ptr;
		if ( !fgets(buf, M, f) || ! strchr(buf, ',')) {
			break;
		}
		// cout << "buf: " << buf << endl;
		// getClass(buf, ',');
		// responses.push_back(buf[0] - 48);
		int res = getClass(buf, ',');
		// cout << "Res about to go in: " << res << endl;
		responses.push_back(res);
		// cout << "buf: " << buf << endl;

		// cout << "responses " << getClass(buf, ',') << " " << endl;
		// ptr = buf + 2;
		int bytes = getBytesToNumber(buf, ',');
		ptr = buf + bytes + 1;

		// cout << "ptr is: " << ptr << endl;

		for (i = 0; i < varCount; i++) {
			int n = 0;
			sscanf(ptr, "%f%n", &elPtr.at<float>(i), &n);
			ptr += n + 1;
		}
		// cout << elPtr << endl;
		if (i < varCount) {
			break;
		}
		_data->push_back(elPtr);
	}
	fclose(f);
	Mat(responses).copyTo(*_responses);

	// printMat("responses", *_responses, (int) 0);
	// printMat("data", *_data, 1.0f);

	return true;
}

template<typename T>
Ptr<T> loadClassifier(const string &loadFile) {
	Ptr<T> model = StatModel::load<T>(loadFile);
	if (model.empty()) {
		cout << "Couldn't read classifier " << loadFile << endl;
	} else {
		cout << "The classifier " << loadFile << " has been loaded" << endl;
	}
	return model;
}

void testAndSaveClassifier(const Ptr<StatModel> &model, const Mat &data, const Mat &responses, 
							int nTrainSamples, int rDelta, const string &saveFile) {
	int i, nSamplesAll = data.rows;
	double trainHr = 0, testHr = 0;
	int trainingCorrectPrediction = 0;

	// Comput prediction error on training data
	for (i = 0; i < nSamplesAll; i++) {
		Mat sample = data.row(i);
		cout << "Sample: " << responses.at<int>(i) << " row " << data.row(i) << endl;
		float r = model->predict(sample);
		cout << "Prediction: " << r << endl;

		if ((int) r == responses.at<int>(i)) {
			trainingCorrectPrediction++;
		}
	}

	printf("Classifier correctly predicted: %i, out of %i sampled\n", trainingCorrectPrediction, nTrainSamples);
	printf("\nTest recognition rate: Training set = %.1f%% \n\n", trainingCorrectPrediction*100.0/nTrainSamples);

	// This function gets called every time so this stuff happens when we're in "Test mode"? When you use the -load arg
	if (saveFile.empty()) {
		// Some stuff 
	}

	if (!saveFile.empty()) {
		model->save(saveFile);
	}
}