# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from setuptools import Extension, find_packages, setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True


setup(
    name='tensorflow-onmttok-ops',
    version='0.1.1',
    description='OpenNMT Tokenizer as TensorFlow Operations',
    author='Emmanuel Ohana',
    author_email='manu.ohana@gmail.com',
    packages=find_packages(),
    install_requires=[
        'tensorflow >= 2.1.0',
    ],
    include_package_data=True,
    ext_modules=[
        Extension('_foo', ['stub.cc'])
    ],
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tensorflow opennmt tokenizer',
)
