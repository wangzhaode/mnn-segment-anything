# Usage

## Compile MNN library
### Linx/Mac
```bash
git clone https://github.com/alibaba/MNN.git
# copy header file
cp -r MNN/include .
cp -r MNN/tools/cv/include .
cd MNN
mkdir build
cmake -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON ..
make -j8
cd ..
cp MNN/build/libMNN.so MNN/build/express/libMNN_Express.so MNN/build/tools/cv/libMNNOpenCV.so ./libs
```

### Windows
```bash
# Visual Studio xxxx Developer Command Prompt
powershell
git clone https://github.com/alibaba/MNN.git
# copy header file
cp -r MNN/include .
cp -r MNN/tools/cv/include .
cd MNN
mkdir build
cmake -G "Ninja" -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON ..
ninja
cd ..
cp MNN.dll MNN.lib ./libs
```

## Build and Run

#### Linux/Mac
```bash
mkdir build && cd build
cmake ..
make -j4
./sam_demo embed.mnn segment.mnn ../../resource/truck.jpg
```
#### Windows
```bash
# Visual Studio xxxx Developer Command Prompt
powershell
mkdir build && cd build
cmake -G "Ninja" ..
ninja
./sam_demo embed.mnn segment.mnn ../../resource/truck.jpg
```
