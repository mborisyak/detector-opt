CC := gcc
PYTHON := python3

# Get Python configuration
PYTHON_INCLUDES := $(shell $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYTHON_LDFLAGS := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LDFLAGS') or '')")
PYTHON_LIBDIR := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_VERSION := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('VERSION'))")
NUMPY_INCLUDES := $(shell $(PYTHON) -c "import numpy; print('-I' + numpy.get_include())")

# Linker flags for macOS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LINK_FLAGS := -undefined dynamic_lookup
else
	LINK_FLAGS := -L$(PYTHON_LIBDIR) -lpython$(PYTHON_VERSION)
endif

detopt/detector/straw_detector.so: detopt/detector/straw_detector.c
	$(CC) -shared -fPIC -Wall -O3 -mtune=native -march=native \
		$(PYTHON_INCLUDES) $(NUMPY_INCLUDES) \
		$(LINK_FLAGS) \
		detopt/detector/straw_detector.c -o detopt/detector/straw_detector.so

all: detopt/detector/straw_detector.so

clean:
	rm -f detopt/detector/straw_detector.so

test: detopt/detector/straw_detector.so
	$(PYTHON) tests/test_solve.py

.PHONY: all clean test
