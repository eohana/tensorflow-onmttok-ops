[![Build Status](https://travis-ci.org/eohana/tensorflow-onmttok-ops.svg?branch=master)](https://travis-ci.org/eohana/tensorflow-onmttok-ops)
[![PyPI version](https://badge.fury.io/py/tensorflow-onmttok-ops.svg)](https://badge.fury.io/py/tensorflow-onmttok-ops)

# OpenNMT Tokenizer TensorFlow Ops

**DISCLAIMER**: This package is not published by the OpenNMT authors.  
Full credits for [OpenNMT Tokenizer](https://github.com/OpenNMT/Tokenizer)
and [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) goes to their respectively
authors.

This project aims to wrap [OpenNMT Tokenizer](https://github.com/OpenNMT/Tokenizer)
into TensorFlow Ops.

It's primarily intended to be used as an addition to the
[OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) framework,
in order to remove the need of applying tokenization and/or 
detokenization outside of a serving environment (e.g. TensorFlow Serving).

## Compatibility

* TensorFlow `2.1`, `2.2`
* OpenNMT-tf >= `2.6.0` *for usage in conjunction with OpenNMT-tf*

## Installation

Prerequisites :

* A Linux environment (`manylinux2014` eligible)
* Python `3.5`, `3.6`, `3.7` or `3.8`

Install the package with pip :

```shell script
pip install tensorflow-onmttok-ops
```

## Usage

### Available Tokenizer options

The majority of the OpenNMT Tokenizer
[options](https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md)
are available.  
However, providing `BPE` or `SentencePiece` models is not supported,
and by extension, setting the tokenizer `mode` to `none` is not supported.

You therefore **cannot** use the following options :

* `bpe_model_path`
* `sp_model_path`
* `sp_nbest_size`
* `sp_alpha`
* `vocabulary_path`
* `vocabulary_threshold`

> **Note:** Tokenizer options are defined at graph construction time
> and are constants.

### Tokenization

```python
import tensorflow_onmttok as tf_onmttok

tokens = tf_onmttok.tokenize(["Hello, how are you?"], mode="conservative")
```

### Detokenization

```python
import tensorflow_onmttok as tf_onmttok

text = tf_onmttok.detokenize(["How", "are", "you", "?"], mode="space")
```

### With OpenNMT-tf

Usage with OpenNMT-tf is pretty straightforward.  
This package comes with a built-in tokenizer 
in order to make usage of the ops.

1. Before training your model, register the tokenizer as follows :

    ```python
    from tensorflow_onmttok import register_opennmt_in_graph_tokenizer
    
    register_opennmt_in_graph_tokenizer()
    ```

    See the [complete example](examples/onmt_tf_training.py)

2. Now that the tokenizer is registered, you can use the 
`OpenNMTInGraphTokenizer` class instead of `OpenNMTTokenizer` in your 
tokenization configuration files, e.g. :

    ```yaml
    type: OpenNMTInGraphTokenizer
    params:
      mode: conservative
      case_feature: true
    ```

3. That's it ! You can now train your model as usual. 
Your `ExportedModel` will now expect a `text` 
input instead of `tokens` and `length`.

    > **Note**: Tokenization resources will not be exported
      to the `assets.extra` directory.

## Build TF Serving with this Ops

This guide will show you how to build TensorFlow Serving
with this ops.

### Prerequisites

* You have already cloned the
TF Serving `>= 2.1.0` [repository](https://github.com/tensorflow/serving),
and have all tools installed for building it
* You have installed CMake `3.1.0` or newer

### Building

#### Add the Ops sources

First, download the 
[release](https://github.com/eohana/tensorflow-onmttok-ops/releases)
of your choice.

Inside the TF Serving sources folder, create a directory
named `custom_ops` and copy the content of the `tensorflow_onmttok`
directory into it.

```shell script
$ cd <tf_serving_sources>
$ mkdir tensorflow_serving/custom_ops
$ cp -r <op_sources>/tensorflow_onmttok tensorflow_serving/custom_ops
```

#### Reference the Ops

Edit `tensorflow_serving/model_servers/BUILD` to reference 
the Ops build target :

```shell script
SUPPORTED_TENSORFLOW_OPS = [
    ...
    "//tensorflow_serving/custom_ops/tensorflow_onmttok:onmttok_ops"
]
```

#### Build OpenNMT Tokenizer from sources

The last step is to build a static version of the
OpenNMT Tokenizer library.  
This repository provides a shell script
that will build it with CMake.

```shell script
$ cd <op_sources>
$ chmod +x build_tokenizer.sh && ./build_tokenizer.sh
```

> **Note**: Pass `sudo` argument to the `build_tokenizer.sh` script
  to execute the `make install` command with sudo.

#### Build TensorFlow Serving

You can now build TensorFlow Serving as usual.
