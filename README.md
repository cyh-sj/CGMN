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


You can also use the following code to evaluate each single model on Flickr30K, MSCOCO 1K and MSCOCO 5K :

```python
from vocab import Vocabulary
import evaluation
evaluation.evalrank("pretrain_model/flickr/model_fliker_1.pth.tar", data_path="$DATA_PATH", split="test", fold5=False)'
evaluation.evalrank("pretrain_model/coco/model_coco_1.pth.tar", data_path="$DATA_PATH", split="testall", fold5=True)'
evaluation.evalrank("pretrain_model/coco/model_coco_1.pth.tar", data_path="$DATA_PATH", split="testall", fold5=False)'
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


## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)


