from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm, trange
import argparse
import numpy as np
import config
torch.multiprocessing.set_start_method('spawn', force=True)
from rawaasist import *

def init():
    parser = config.initParams()
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")


    return args


def source_tracing_id(feat_model_path):
    checkpoint = torch.load(feat_model_path)
    if args.model == 'W2VAASIST':
        model = XLSRAASIST(model_dir=args.SSL_path).cuda()
    if args.model == 'Rawaasist':
        model = Rawaasist().cuda()
    if args.model == 'mellcnn':
        model = MELLCNN().cuda()
        
    model.load_state_dict(checkpoint)
    model.eval()
    evalfeas, evallogits, evallabels = [], [], []

    test_set = codecfake_st_eval(path_to_features= args.eval_wav, path_to_protocol= args.eval_label)
                                 
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    with open('./result/result_id.txt', 'w') as f:
        for idx, data_slice in enumerate(tqdm(testDataLoader)):
            waveform, filenames,labels = data_slice[0],data_slice[1],data_slice[2] 
            evallabels.append(torch.tensor(labels).cpu())      
            with torch.no_grad(): 
                feats, logits = model(waveform)
            
            scores = torch.nn.functional.softmax(logits, dim=1)
            vectors, predicted = torch.max(scores, dim=1)
            for i in range(feats.shape[0]):
                filename = filenames[i]  
                pred_label = int(predicted[i].item())  
                score = vectors[i].item()  
                f.write(f'{filename} {pred_label} {score}\n') 

            evallogits.append(logits)
            evalfeas.append(feats)

    evallabels = torch.cat(evallabels, dim=0)
    evalfeas = torch.cat(evalfeas, dim=0)
    evallogits = torch.cat(evallogits, dim=0)

    torch.save({"feas": evalfeas, "logits": evallogits, "labels": evallabels}, './save_feats/evaldict_id.pt')   # save feats,logits,label


if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    
    source_tracing_id(model_path)