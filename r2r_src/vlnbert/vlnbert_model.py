import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import _pickle as cPickle

import sys
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

# from vlnbert.modeling_bert import LanguageBert
from vlnbert.modeling_visbert import VLNBert

model_name_or_path = 'load/hop/pytorch_model.bin'

def get_tokenizer():
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
    return tokenizer

def get_vlnbert_models(config=None):
    config_class = BertConfig
    model_class = VLNBert
    vis_config = config_class.from_pretrained('bert-base-uncased')

    # all configurations (need to pack into args)
    vis_config.img_feature_dim = 2176
    vis_config.img_feature_type = ""
    vis_config.update_lang_bert = False
    vis_config.update_add_layer = False
    vis_config.vl_layers = 4
    vis_config.la_layers = 9
    visual_model = VLNBert(vis_config)

    visual_model = model_class.from_pretrained(model_name_or_path, config=vis_config)

    return visual_model
