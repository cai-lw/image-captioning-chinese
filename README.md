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

Images are from MS COCO. To save time from running huge CNNs, they are provided as feature vectors from a pre-trained CNN. To prevent cheating (manual solving), only a small fraction of the original images are provided.

Captions are labeled by students in the course, so they may not be high-quality.

The dataset can be downloaded at [Google Drive](https://drive.google.com/drive/u/1/folders/13EiOI11_Hg3S2vJDU5oRGa1kyT01h7KZ) or [百度网盘](https://pan.baidu.com/s/1LDHc6Fx7VHR4zhkzdRRc7Q).

## Usage

[Download METEOR](http://www.cs.cmu.edu/~alavie/METEOR/) and put it in directory `meteor-1.5`, and run `make_val_meteor.py` to produce METEOR-compatible validation data.

`lstm.py ` is the "Show and Tell" model，and `lstm_attention.py` is the "Show, Attend and Tell" model. Both models have many configurable hyperparameters. Run them with `--help` argument to learn more.
