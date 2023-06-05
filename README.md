# Cross-Modal Graph Matching Network for Image-Text Retrieval (CGMN)
PyTorch code for CGMN described in the paper "Cross-Modal Graph Matching Network for Image-text Retrieval". The paper is accepted by Transactions on Multimedia Computing Communications and Applications. It is built on top of the [VSE++](https://github.com/fartashf/vsepp).

Partial data can be obtained [here](https://drive.google.com/file/d/1ZVLIN7uSh3dqYAEldelyYF2ei9vicJvZ/view?usp=sharing), and the pretrained models can be obtained in [Flickr30K](https://drive.google.com/file/d/12FO57QvTetKB8ex7kxhS3yhHaUB1k9Mp/view?usp=sharing) and [MS-COCO](https://drive.google.com/file/d/1N10l7mkeQ7R-KOAARe9ZXaQZRS8iWHwH/view?usp=sharing).

The IOU.npy can be obtained by using getiou.py with _bbx.npy .

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



## Reference

If you found this code useful, please cite the following paper:

```
@article{Cheng2022CGMN,
author = {Cheng, Yuhao and Zhu, Xiaoguang and Qian, Jiuchao and Wen, Fei and Liu, Peilin},
title = {Cross-Modal Graph Matching Network for Image-Text Retrieval},
year = {2022},
issue_date = {November 2022},
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
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)


