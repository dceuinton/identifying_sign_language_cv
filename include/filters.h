#ifndef DALES_FILTERS_A2
#define DALES_FILTERS_A2

#include <opencv2/opencv.hpp>

using namespace cv;

const int THRESHOLD_BINARY = 0;
const int THRESHOLD_OTSU   = 1;

void displayImage(Mat *src, const char *windowName);
void filterToGreyScale(Mat *src);
void filterToOnlyBlack(Mat *src);
void filterToOnlyBlue(Mat *src);
void medianFilter(Mat* src);
Mat* threshold(Mat *src, int thresh, int threshType, bool show);
void otsu(Mat *src, int thresh, bool show);

#endif