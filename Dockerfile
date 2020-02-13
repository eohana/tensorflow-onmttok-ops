ARG PYTHON_VERSION=3.6
ARG OPENNMT_TOKENIZER_COMMIT_SHA="08ba3951c9093603ca2b58bd3a6fa45e9c8fb22f"

FROM python:$PYTHON_VERSION

ARG OPENNMT_TOKENIZER_COMMIT_SHA

RUN apt-get update && \
    apt-get install -y git rsync gcc g++ cmake

# Build OpenNMT Tokenizer
WORKDIR /opt

RUN git clone https://github.com/OpenNMT/Tokenizer.git /opt/opennmt-tokenizer && \
    cd /opt/opennmt-tokenizer && \
    git checkout ${OPENNMT_TOKENIZER_COMMIT_SHA} && \
    mkdir build && cd build && \
    cmake -DCMAKE_CXX_FLAGS=-fPIC \
          -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
          -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=true \
          -DCMAKE_BUILD_TYPE=Release \
          -DLIB_ONLY=ON \
          -DBUILD_SHARED_LIBS=OFF .. && \
    make install

# Install TensorFlow
RUN pip install --upgrade pip && \
    pip install tensorflow

# Build the Op
ADD . /custom_op

WORKDIR /custom_op

RUN make tf_onmttok_pip_pkg tf_onmttok_test
