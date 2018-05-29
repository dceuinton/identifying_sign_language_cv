
#include "filters.h"

void displayImage(Mat *src, const char *windowName) {
	imshow(windowName, *src);
	waitKey(0);
}

void filterToGreyScale(Mat *src) {
	cvtColor(*src, *src, CV_BGR2GRAY);
}

void filterToOnlyBlack(Mat *src) {
	int threshold = 22;
	filterToGreyScale(src);
	uchar *rowPtr;
	for (int i = 0; i < src->rows; i++) {
		rowPtr = src->ptr<uchar>(i);
		for (int j =0; j < src->cols; j++) {
			if (rowPtr[j] < threshold) {
				rowPtr[j] = 0;
			} else {
				rowPtr[j] = 255;
			}
		}
	}
}

void filterToOnlyBlue(Mat *src) {
	uchar *rowPtr;
	int channels = src->channels();
	int rows = src->rows;
	int cols = src->cols;

	for (int i = 0; i < rows; i++) {
		rowPtr = src->ptr<uchar>(i);
		for (int j =0; j < cols; j++) {
			if (rowPtr[j * channels] < 150) {
				rowPtr[j * channels] = 255;
				rowPtr[j * channels + 1] = 255;
				rowPtr[j * channels + 2] = 255;
			} else if (rowPtr[j * channels + 1] > 100) {
				rowPtr[j * channels + 0] = 255;
				rowPtr[j * channels + 1] = 255;
				rowPtr[j * channels + 2] = 255;
			} else if (rowPtr[j * channels + 2] > 100) {
				rowPtr[j * channels + 0] = 255;
				rowPtr[j * channels + 1] = 255;
				rowPtr[j * channels + 2] = 255;
			}
		}
	}
}

// void filterOutBlue(Mat *src) {
// 	cvtColor(*src, *src, COLOR_BGR2HSV);
// }

void medianFilter(Mat* src) {
	medianBlur(*src, *src, 3);
}

const char *binaryTrackbar = "Binary Threshold";
const char *otsuTrackbar   = "Otsu Threshold";
const char *binaryWindow   = "Binary Thresholding";
const char *otsuWindow     = "Otsu Thresholding";
const char *actualWindow;

Mat thresholdImage, thresholdImageFinal;

int thresholdClick = 0;
int thresholdType = 0;

void thresholdTrackbarCallback(int value, void *object) {
	// Mat image(thresholdImage->size(), CV_8UC1);
	if (thresholdType == THRESHOLD_BINARY) {
		threshold(thresholdImage, thresholdImageFinal, thresholdClick, 255, THRESH_BINARY);	
	} else if (thresholdType == THRESHOLD_OTSU) {
		threshold(thresholdImage, thresholdImageFinal, thresholdClick, 255, THRESH_OTSU);
	} else {
		printf("ERROR: No valid threshold type detected. Check filters.h for types.\n");
		return;
	}
	imshow(actualWindow, thresholdImageFinal);
}

Mat* threshold(Mat *src, int thresh, int threshType, bool show) {

	Mat *output = new Mat(src->size(), src->type());
	*output = *src;

	if (src->type() != CV_8UC1) {
		filterToGreyScale(output);
	}

	const char *trackbar;

	thresholdImage = *output;
	thresholdType = threshType;
	thresholdClick = thresh;

	if (show) {
		if (thresholdType == THRESHOLD_BINARY) {
			actualWindow = binaryWindow;
			trackbar = binaryTrackbar;
			threshold(thresholdImage, thresholdImageFinal, thresh, 255, THRESHOLD_BINARY);
		} else if (thresholdType == THRESHOLD_OTSU) {
			actualWindow = otsuWindow;
			trackbar = otsuTrackbar;
			threshold(thresholdImage, thresholdImageFinal, thresh, 255, THRESHOLD_OTSU);
		}

		namedWindow(actualWindow, WINDOW_AUTOSIZE);
		createTrackbar(trackbar, actualWindow, &thresholdClick, 255, thresholdTrackbarCallback);
		setTrackbarPos(trackbar, actualWindow, thresholdClick);

		imshow(actualWindow, thresholdImageFinal);
		waitKey(0);
	}

	if (thresholdType == THRESHOLD_BINARY) {
		// printf("Click is: %i\n", thresholdClick);
		threshold(thresholdImage, *output, thresholdClick, 255, THRESH_BINARY);
	} else if (thresholdType == THRESHOLD_OTSU) {
		// printf("Click is: %i\n", thresholdClick);
		threshold(thresholdImage, *output, thresholdClick, 255, THRESH_OTSU);
	}

	return output;
}

void otsu(Mat *src, int thresh, bool show) {
	if (src->type() != CV_8UC1) {
		filterToGreyScale(src);
	}
	threshold(*src, *src, thresh, 255, THRESH_OTSU);
}



