CC := gcc
PYTHON := python

detopt/detector/straw_detector.so: detopt/detector/straw_detector.c
	$(CC) -shared -fPIC -Wall -O3 -mtune=native -march=native \
		$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('CFLAGS'))") \
		-I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))") \
		-I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())") \
		-L$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
		$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LDFLAGS'))") \
		-lpython$(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") \
		detopt/detector/straw_detector.c -o detopt/detector/straw_detector.so

all: detopt/detector/straw_detector.so

test: detopt/detector/straw_detector.so
	python tests/test_solve.py
