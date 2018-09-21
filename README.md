## Installations for running keras-theano on GPU
1. Upgrade pip and install opencv2
```
cd ~
pip install --upgrade pip
pip install opencv-python
```
2. Upgrade keras and set theano background
```
pip uninstall keras
pip install keras
vi ~/.keras/keras.json
  {
    "backend": "theano",
    "image_data_format": "channels_first",
    "floatx": "float32",
    "epsilon": 1e-07
  }
```
3. Upgrade theano
```
pip uninstall theano
pip install theano==1.0.1
```
4. Install pygpu which is necessary to run theano on GPU
```
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
mkdir Build
cd Build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
make install
cd ..
python3 setup.py build_ext -L /root/libgpuarray/lib -I /root/libgpuarray/include
python3 setup.py install
sudo ldconfig
```
5. Run python on a specific GPU
```
THEANO_FLAGS=device=cuda0 python3
```

## How to run the project
### Training
1. Download [RASM2018 example dataset](https://www.dropbox.com/s/j4348fx4k7ow4zh/RASM2018_Example_Set.zip?dl=0)

2. Create labeled images
```
python3 parse_data.py
```
3. Create train patches
```
python3 TrainPatchMaker.py
```
4. Train FCN and save the best weights
```
python3 train.py
```
### Testing
1. Download [RASM2018 evaluation dataset](https://www.primaresearch.org/RASM2018/)

2. Run PageSegmentation.py to predict page segmentations of evaluation set.
```
THEANO_FLAGS=device=cuda0 python3 PageSegmentation.py
```
