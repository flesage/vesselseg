install
-------

``git clone --recurse-submodules https://github.com/flesage/vesselseg.git``

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

next
----
add online data augmentation
https://pythonawesome.com/tools-for-augmenting-and-writing-3d-medical-images-on-pytorch/
