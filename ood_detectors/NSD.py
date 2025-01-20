import torch
from typing import Dict

from ood_detectors.interface import OODDetector
import numpy as np
# normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
import torch.nn.functional as F
from copy import deepcopy



def NSD_with_angle(feas_train, feas, min=False):

    cos_similarity = torch.mm(feas, feas_train.T)
    
    if min:
        scores = cos_similarity.min(dim=1).values
    else:
        scores = cos_similarity.mean(dim=1)
        
    return scores


class NSDOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        logits_train = train_model_outputs['logits']
        feas_train = train_model_outputs['feas']
        train_labels = train_model_outputs['labels']
        feas_train = F.normalize(feas_train, p=2, dim=-1)
        confs_train = torch.logsumexp(logits_train, dim=1)
        self.scaled_feas_train = feas_train * confs_train[:, None]

    def infer(self, model_outputs):
        feas = model_outputs['feas']
        logits = model_outputs['logits']

        # Calculate cosine similarity
        feas = F.normalize(feas, p=2, dim=-1)
        confs = torch.logsumexp(logits, dim=1)

        guidances = NSD_with_angle(self.scaled_feas_train, feas)
        scores = guidances.to(confs.device) * confs
        return scores
