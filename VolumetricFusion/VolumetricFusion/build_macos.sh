# !/bin/zsh

#rm -r CMakeCache.txt CMakeFiles/ ; cmake ../VolumetricFusion VolumetricFusion && make -j4 ../VolumetricFusion

rm -r CMakeCache.txt CMakeFiles ; cmake . && make -j4
