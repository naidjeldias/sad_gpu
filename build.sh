#!/bin/bash
set -e

mkdir -p build/
cd build/
cmake -D WITH_DISPLAY=OFF ..
thread_num=$(nproc)
make -j$((++thread_num))
cd ..
