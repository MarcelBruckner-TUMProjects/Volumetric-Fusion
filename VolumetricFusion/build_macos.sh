# !/bin/zsh

rm -r CMakeCache.txt CMakeFiles/ ; cmake . && make -j4 VolumetricFusion
