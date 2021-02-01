===============
VesselSeg 
===============
Pytorch implementation of various VNet like models for 3D vascular segmentation from microscopy angiograms 

install
-------
``git clone --recurse-submodules https://github.com/flesage/vesselseg.git``

``cd vesselseg``

``conda create -n ${ENV_NAME} python=3.7 pytorch spyder scipy scikit-image scikit-learn matplotlib``

``source activate ${ENV_NAME}``

``pip install h5py tqdm``

``cd Magic_VNet``

``python setup.py install``

``cd ..``

intstall optional
-----------------
``pip install --upgrade https://github.com/VincentStimper/mclahe/archive/numpy.zip``

``sudo apt-get install nvidia-cuda-toolkit``

training
--------
training using different hyper parameter: vnet model choice, number of epochs, loss function, learning rate, etc, can be done using 'train.py'

the out model along with training/val/testing cureves will saved to 'results/' with unique datetime stamp. 


prediction
----------
example to run prediction:
``python predict.py -i im_test.tif -m results/VNet_BSC_31_01_2021_14_51 -k 64 64 64``

for the available arguments:
``python predict.py -h``

to do
-----
add online data augmentation
https://pythonawesome.com/tools-for-augmenting-and-writing-3d-medical-images-on-pytorch/

- different batch size
- rotation 
- gaussian noise
- inhomogeneity 
