CC := gcc

detopt/detectors/straw_detector.so: detopt/detectors/straw_detector.c
	$(CC) --shared -fPIC -Wall -O3 -mtune=native -march=native \
		$(shell python-config --includes) $(shell python-config --libs) $(shell numpy-config --cflags) \
		detopt/detectors/straw_detector.c -o detopt/detectors/straw_detector.so

all: detopt/detectors/straw_detector.so

test: detopt/detectors/straw_detector.so
	python tests/test_solve.py