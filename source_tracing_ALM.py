from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm, trange
import argparse

from torch.utils.data import ConcatDataset
import config
torch.multiprocessing.set_start_method('spawn', force=True)

from rawaasist import *

def init():
    base_args = config.initParams()
    parser = argparse.ArgumentParser("load model scores")

    for arg in vars(base_args):
        parser.add_argument(f'--{arg}', default=getattr(base_args, arg))

    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")


    return args


def source_tracing_ALM(feat_model_path):
    # Load the model checkpoint
    checkpoint = torch.load(feat_model_path)
    
    # Select the model based on the argument
    if args.model == 'W2VAASIST':
        model = XLSRAASIST(model_dir=args.SSL_path).cuda()
    elif args.model == 'Rawaasist':
        model = Rawaasist().cuda()
    elif args.model == 'mellcnn':
        model = MELLCNN().cuda()
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Define your test sets and corresponding output files
    test_sets = {
        "mini_omni": ood_alm(audio_folder_path=args.ALM_mini_omni),
        "moshi": ood_alm(audio_folder_path=args.ALM_moshi),
        "valle": ood_alm(audio_folder_path=args.ALM_valle),
        "speechgpt": ood_alm(audio_folder_path=args.ALM_speechgpt)
    }

    output_files = {
        "mini_omni": './result/alm/miniomni.txt',
        "moshi": './result/alm/moshi.txt',
        "valle": './result/alm/valle.txt',
        "speechgpt": './result/alm/speechgpt.txt'
    }

    # Process each test set
    for key in test_sets:
        test_set = test_sets[key]
        output_file = output_files[key]
        
        testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

        with open(output_file, 'w') as f:
            for idx, data_slice in enumerate(tqdm(testDataLoader)):
                waveform, filenames = data_slice[0], data_slice[1]
                with torch.no_grad(): 
                    feats, logits = model(waveform)
                
                scores = torch.nn.functional.softmax(logits, dim=1)
                vectors, predicted = torch.max(scores, dim=1)
                for i in range(feats.shape[0]):
                    filename = filenames[i]  
                    pred_label = int(predicted[i].item())  
                    score = vectors[i].item()  
                    f.write(f'{filename} {pred_label} {score}\n') 



if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    source_tracing_ALM(model_path)