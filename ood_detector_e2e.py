from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
import pickle
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from ood_detectors.factory import create_ood_detector
import eval_metrics as em
from scipy import interpolate
from sklearn.metrics import confusion_matrix, f1_score,roc_curve,roc_auc_score
import numpy as np


def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    parser.add_argument("--ood_detector_name", type=str, default="msp",
                        choices=['energy', 'msp', 'NSD',  'mahalanobis',  'knn'])
    parser.add_argument("--condition", type=str, default="single7",
                    choices=['single7','multi8','voc','diff_config','diff_source'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.out_score_dir = "./scores" if '19' in args.task else args.score_dir

    return args

def load_dict(dict_path):
    with open(dict_path, 'rb') as f:
        return pickle.load(f)

def save_dict(dict_obj, dict_path):
    with open(dict_path, 'wb') as f:
        pickle.dump(dict_obj, f)

def test_ood():
    traindict = torch.load('./save_feats/traindict.pt')

    # id are 0-6, ood7 if fake 7, ood 8 are fake 8-11, oodvoc are 19fake and ITWfake.
    evaldictid = torch.load('./save_feats/evaldict_id.pt')
    if args.condition == 'single7':
        evaldictood = torch.load('./save_feats/evaldict_ood7.pt')
    if args.condition == 'multi8':
        evaldictood = torch.load('./save_feats/evaldict_ood8.pt')
    if args.condition == 'voc':
        evaldictood = torch.load('./save_feats/evaldict_oodvoc.pt')
    if args.condition == 'diff_config':
        evaldictood = torch.load('./save_feats/evaldict_ood_config.pt')
    if args.condition == 'diff_source':
        evaldictood = torch.load('./save_feats/evaldict_ood_source.pt')
    evaldict = {}
    evaldict["logits"] = torch.cat([evaldictid["logits"], evaldictood["logits"]], dim=0)
    evaldict["feas"] = torch.cat([evaldictid["feas"], evaldictood["feas"]], dim=0)
    evaldict["labels"] = torch.cat([evaldictid["labels"], evaldictood["labels"]], dim=0)
    
    print(evaldictid["logits"].shape, 'evaldictid')
    print(evaldictid["feas"].shape, 'evaldictid')
    print(evaldictid["labels"].shape, 'evaldictid')
    print(evaldictood["logits"].shape, 'evaldictood')
    print(evaldictood["feas"].shape, 'evaldictood')
    print(evaldictood["labels"].shape, 'evaldictood')
    print(evaldict["logits"].shape, 'evaldict')
    print(evaldict["feas"].shape, 'evaldict')
    print(evaldict["labels"].shape, 'evaldict')
    for key in traindict:
        traindict[key] = traindict[key].cpu()

    # Executing OOD detector
    print("Running OOD detector...")
    saved_detector_path = "./ood_step/detector.pkl"
    ood_detector = create_ood_detector(args.ood_detector_name)
    ood_detector.setup(args, traindict)

    print("Evaluating metrics...")
    for key in evaldict:
        evaldict[key] = evaldict[key].cpu()

    predicted_logits = ood_detector.infer(evaldict)

    # torch.save(id_scores, './ood_step/result_oodscore.pt')

    true_labels = evaldict["labels"]

    binary_true_labels = [0 if label in [7, 8, 9, 10, 11] else 1 for label in true_labels]
        
    auc_score = roc_auc_score(binary_true_labels, predicted_logits)
    fpr, tpr, thresholds = roc_curve(binary_true_labels, predicted_logits)
        
    idx_95_tpr = (tpr >= 0.95).nonzero()[0][0] 
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))

    target_scores = np.array([predicted_logits[i] for i in range(len(binary_true_labels)) if binary_true_labels[i] == 1])
    nontarget_scores = np.array([predicted_logits[i] for i in range(len(binary_true_labels)) if binary_true_labels[i] == 0])

    eer, eer_threshold = em.compute_eer(target_scores, nontarget_scores)

    print(f"auc_score: {auc_score:.4f}")
    print(f"fpr95: {fpr95:.4f}")
    print(f"eer_cm: {eer:.4f}")





if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    test_ood()
