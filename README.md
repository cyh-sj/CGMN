# Cross-Modal Graph Matching Network for Image-Text Retrieval (CGMN)
PyTorch code for CGMN described in the paper "Cross-Modal Graph Matching Network for Image-text Retrieval". The paper is accepted by Transactions on Multimedia Computing Communications and Applications. It is built on top of the [VSE++](https://github.com/fartashf/vsepp).

## Requirements 
We recommended the following dependencies.

* Python 2.7 
* [PyTorch](http://pytorch.org/) (0.4.1)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [torchvision]()
* [matplotlib]()


* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Evaluate pre-trained models
Modify the model_path and data_path in the evaluation_models.py file. Then Run `evaluation_models.py`:

```bash
python evaluation_models.py
```

## Training new models
Run `train.py`:

For MSCOCO:

```bash
python train.py --data_path $DATA_PATH --data_name coco_precomp --logger_name runs/coco_VSRN --max_violation
```

For Flickr30K:

```bash
python train.py --data_path $DATA_PATH --data_name f30k_precomp --logger_name runs/flickr_CGMN --max_violation --lr_update 10  --max_len 60
```

The pretrained models will be released soon.

## Reference


If you found this code useful, please cite the following paper:
'''
@article{10.1145/3499027,
author = {Cheng, Yuhao and Zhu, Xiaoguang and Qian, Jiuchao and Wen, Fei and Liu, Peilin},
title = {Cross-Modal Graph Matching Network for Image-Text Retrieval},
year = {2022},
issue_date = {November 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {18},
number = {4},
issn = {1551-6857},
url = {https://doi.org/10.1145/3499027},
doi = {10.1145/3499027},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
month = {mar},
articleno = {95},
numpages = {23},
keywords = {Image-text retrieval, relation reasoning, cross-modal matching, graph matching}
}
'''

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)


