
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
	// otsu(src, 10);
	threshold(src, 10, THRESHOLD_BINARY, true);
	// threshold(src, 10, THRESHOLD_OTSU, true);

	displayImage(src, "Testing");

	return 0;
}