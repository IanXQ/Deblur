# Deblur
This project implements **ACDeblerNet** of paper *Multi-Scale Convolutional Neural Network for Image Deblurring* (ChinaMM2020).
# Overview
Image deblurring is one of the hot topics in image processing community. Conventional image deblurring algorithms are weak in predicting fuzzy kernel properly, which further leads to unreliable deblurred results. To solve above mentioned issues, this paper proposes a method based on multi-scale convolutional neural network. Firstly, this method combines the features of different scales of the image to fully extract the image information. Secondly, introducing the modified parameter reuse strategy, that is, sharing the parameters between different scales to shrink the solution space, and selectively sharing the parameters within the same scale to reduce the model parameters. In addition, combined with the attention mechanism, designing the attention convolution group, which gives rise to expanding the receptive field. Experimental results show that this method effectively improves the network performance. Moreover, the restored images have good visual deblurring quality.
# Environment
This experiments is implemented with Pytorch. We tested the codes with Pytorch 1.4.0 GPU version, CUDA 10.0 and Python 3.6.12 on Ubuntu 18.04.5 with GTX 1080Ti.

