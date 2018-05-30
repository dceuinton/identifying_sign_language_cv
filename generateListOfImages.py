#!/usr/bin/env python

import sys, os.path

if __name__ == "__main__":

	IMAGE_DIR = "./gestures/";

	if len(sys.argv) == 2:
		IMAGE_DIR = sys.argv[1];

	count = 0;

	for dirname, dirnames, filenames in os.walk(IMAGE_DIR):
		for filename in filenames:
			path = os.path.join(dirname, filename);
			print path
			count += 1;

	print "Number of images: %i" % (count);
			