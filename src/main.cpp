
#include <stdio.h>
#include <string>

#include "filters.h"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[]) {
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
	displayImage(binaryImage, "Filtered");

	// Finding contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(*binaryImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Drawing Contours
	int idx = 0;
	for (; idx >=0; idx = hierarchy[idx][0]) {
		Scalar colour(rand() & 255, rand() & 255, rand() & 255);
		drawContours(copy, contours, idx, colour, 1, 8, hierarchy);
	}

	// Display image with contours
	displayImage(&copy, "Contours");

	// Displaying original image
	displayImage(src, "Original");

	delete src;
	src = NULL;
	delete binaryImage;
	binaryImage = NULL;

	return 0;
}

// void ellipticFourierDescriptors(vector<Point> &contour, vector<float> )