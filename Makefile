CXX := g++
PYTHON_BIN_PATH = python

TF_ONMTTOK_SRCS = $(wildcard tensorflow_onmttok/cc/kernels/*.cc) $(wildcard tensorflow_onmttok/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -Wl,--no-undefined -shared ${TF_LFLAGS} -lOpenNMTTokenizer -pthread

TF_ONMTTOK_TARGET_LIB = tensorflow_onmttok/python/ops/_tensorflow_onmttok_ops.so

tf_onmttok_op: $(TF_ONMTTOK_TARGET_LIB)

$(TF_ONMTTOK_TARGET_LIB): $(TF_ONMTTOK_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

tf_onmttok_test: tensorflow_onmttok/python/ops/onmttok_ops_test.py tensorflow_onmttok/python/ops/onmttok_ops.py $(TF_ONMTTOK_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_onmttok/python/ops/onmttok_ops_test.py

tf_onmttok_pip_pkg: $(TF_ONMTTOK_TARGET_LIB)
	sudo ./build_pip_pkg.sh make artifacts
