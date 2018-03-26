# Image Captioning in Chinese

A course project of *Pattern Recognition* at Tsinghua University in the spring semester of 2017.

Implemented two RNN-based image captioning models from two corresponding papers:

* "Show and Tell", simple LSTM RNN: *Vinyals, Oriol, et al. "Show and tell: Lessons learned from the 2015 MSCOCO image captioning challenge."*
* "Show, Attend and Tell", LSTM RNN with attention: *Xu, K., et al. "Show, attend and tell: Neural image caption generation with visual attention."*

## Dependencies

* tensorflow 1.1
* tensorlayer 1.4.3
* jieba 0.38
* h5py 2.7.0

## Dataset

Images are from MS COCO. To save time from running huge CNNs, they are provided as feature vectors from a pre-trained CNN.

Captions are labeled by students in the course.

The dataset is too large (>1GB) and not provided here. Contact me if you want the data.

## Usage

[Download METEOR](http://www.cs.cmu.edu/~alavie/METEOR/) and put it in directory `meteor-1.5`, and run `make_val_meteor.py` to produce METEOR-compatible validation data.

`lstm.py ` is the "Show and Tell" model，and `lstm_attention.py` is the "Show, Attend and Tell" model. Both models have many configurable hyperparameters. Run them with `--help` argument to learn more.