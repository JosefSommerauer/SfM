# SfM

## Build
```
mkdir build
cd build
cmake .. 
make
```

## Run
```
./build/SFM \
	-i <input_list.txt> \
	-k <camera calibration.xml> \ 
	-f AKAZE  # ORB, ORBGPU, ORB5000, ORB5000GPU, SURF, SIFT, KAZE, GPU (SURF GPU)
 ```
