
#include <stdio.h>
#include <string>
#include <cmath>
#include <fstream>

#include "filters.h"

using namespace std;
using namespace cv;

const char *descriptorFilename = "descriptors.data";

void ellipticFourierDescriptors(vector<Point> &contour, vector<float> &CE);
void writeDescriptors(const char *filename, vector<float> &CE);
void clearContentsOfFile(const char *filename);

int main(int argc, char const *argv[]) {
	// Print version for my own knowledge (had to update it earlier)
	printf("OpenCV Version %i.%i\n", CV_MAJOR_VERSION, CV_MINOR_VERSION);

	const char *filename;

	if (argc == 2) {
		filename = argv[1];
		printf("Opening %s\n", filename);		
	} else {
		printf("Usage: main <filename>\n");
		return 1;
	}

	Mat *src = new Mat();
	*src = imread(filename);
	Mat copy(src->size(), src->type(), Scalar(0,0,0));

	// Binarizing Image
	Mat *binaryImage = threshold(src, 2, THRESHOLD_BINARY, false);
	// displayImage(binaryImage, "Filtered");

	// Finding contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(*binaryImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// printf("Contours size: %i\n", (int)contours.size());

	// Drawing Contours
	int idx = 0;
	for (; idx >=0; idx = hierarchy[idx][0]) {
		Scalar colour(rand() & 255, rand() & 255, rand() & 255);
		drawContours(copy, contours, idx, colour, 1, 8, hierarchy);
		// imshow("Step", copy);
		// waitKey(0);
	}

	// Get Elliptical Fourier Descriptors
	vector<float> CE;
	ellipticFourierDescriptors(contours[0], CE);
	// for (int i = 0; i < CE.size(); i++) {
	// 	printf("%i: %f\n", (i+1), CE[i]);
	// }

	// File writing things
	clearContentsOfFile(descriptorFilename);
	// writeDescriptors(descriptorFilename, CE);
	// writeDescriptors(descriptorFilename, CE);

	// // Display image with contours
	// displayImage(&copy, "Contours");

	// // Displaying original image
	// displayImage(src, "Original");

	delete src;
	src = NULL;
	delete binaryImage;
	binaryImage = NULL;

	return 0;
}

// The C implementation that I borrowed from the lecture slides and notes. 
void ellipticFourierDescriptors(vector<Point> &contour, vector<float> &CE) {
	vector<float> ax, ay, bx, by;
	int m = contour.size();
	int n = 20;                        // Number of CEs 
	float t = (2*M_PI)/m;

	for (int k = 0; k < n; k++) {
		ax.push_back(0.0);    ay.push_back(0.0);
		bx.push_back(0.0);    by.push_back(0.0);

		for (int i = 0; i < m; i++) {
			ax[k] = ax[k] + contour[i].x * cos((k + 1) * i * t);
			ay[k] = ay[k] + contour[i].y * cos((k + 1) * i * t);
			bx[k] = bx[k] + contour[i].x * sin((k + 1) * i * t);
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

void clearContentsOfFile(const char *filename) {
	ofstream file(filename, ofstream::out | ofstream::trunc);
	file.close();
}

void writeDescriptors(const char *filename, vector<float> &CE) {
	ofstream file(filename, ofstream::out | ofstream::app);

	for (int i = 0; i < CE.size(); i++) {
		file << CE[i] << " ";
	}
	file << endl;

	file.close();
}