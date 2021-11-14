import torch
from vocab import Vocabulary
import evaluation
import evaluation_models

#torch.cuda.set_device(1)
evaluation_models.evalrank(path1, path2, data_path=data_path, split="test", fold5=False)
