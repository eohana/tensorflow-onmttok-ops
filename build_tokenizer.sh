#!/usr/bin/env bash
TOKENIZER_VERSION=1.18.1

if [[ ${1} == "sudo" ]]; then
  MAKE_CMD_PREFIX=${1}
else
  MAKE_CMD_PREFIX=""
fi

cd /opt
wget https://github.com/OpenNMT/Tokenizer/archive/v$TOKENIZER_VERSION.tar.gz
tar -xf v$TOKENIZER_VERSION.tar.gz && cd Tokenizer-$TOKENIZER_VERSION
mkdir build && cd build

cmake -DCMAKE_CXX_FLAGS=-fPIC \
      -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=true \
      -DCMAKE_BUILD_TYPE=Release \
      -DLIB_ONLY=ON \
      -DBUILD_SHARED_LIBS=OFF ..

$MAKE_CMD_PREFIX make install
