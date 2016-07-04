Cross-Domain-Landmarks-Selection-CDLS-
=========================
##Author: Yao-Hung (Hubert) Tsai <yaohungt@andrew.cmu.edu>

#####Package with code and demo usage for the paper:</br>
#####"Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation"</br>
#####    Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang</br>
#####    Computer Vision and Pattern Recognition (CVPR) 2016.

Setup:
------
- Dowload and compile libsvm with weights support
    - The path to **libsvm-weights** code available at
        <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/weights/libsvm-weights-3.20.zip>
- Edit **CDLS_demo.m** with desired parameters and path
    - edit the path to **/libsvm-weights/matlab**
    - edit the paramters of *iter*, *delta*, and *PCA_dimension* if you want
- Put you own data in **/data**
    - E.g., **/data/amazon_DeCAF_dslr_SURF.mat** is a random split of images from _Office and Caltech-256 Datasets_ (Amazon images with _DeCAF_ features and DSLR images with _SURF_ features)

Run:
-----
- Directly run **CDLS_demo.m**
