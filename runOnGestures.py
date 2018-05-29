#!/usr/bin/env python

import sys, os.path, subprocess

if __name__ == "__main__":

	IMAGE_DIR = "./gestures/";
	count = 0;
	limit = 10;

	if len(sys.argv) == 2:
		limit = int(sys.argv[1]);
		print limit;

	for dirname, dirnames, filenames in os.walk(IMAGE_DIR):
		for file in filenames:
			if (count < limit):
				filepath = os.path.join(IMAGE_DIR, file);
				print filepath;
				print count;
				subprocess.call(["./main", filepath]);
				count += 1;

			