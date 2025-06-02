CC := gcc

detopt/detector/straw_detector.so: detopt/detector/straw_detector.c
	$(CC) --shared -fPIC -Wall -O3 -mtune=native -march=native \
		$(shell python-config --includes) $(shell python-config --libs) $(shell numpy-config --cflags) \
		detopt/detector/straw_detector.c -o detopt/detector/straw_detector.so

all: detopt/detector/straw_detector.so

test: detopt/detector/straw_detector.so
	python tests/test_solve.py