import re
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import argparse

with open("./Data/CATAMARAN.json", 'r', encoding='utf-8') as f, open("./Data/CATAMARAN_V1_Samples.json", 'w',
                                                                     encoding='utf-8') as fp:
    res = {"Description": "This is a sample dataset containing 5000 pieces of data.", "Data": []}
    data = f.readlines()

    for i, line in enumerate(data):
        if i > 5000: break
        dec_json = json.loads(line)
        res["Data"].append(dec_json)
    json.dump(res, fp, ensure_ascii=False, indent=4)
