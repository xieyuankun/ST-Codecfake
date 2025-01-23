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



def source_tracing_id_source(feat_model_path):
    checkpoint = torch.load(feat_model_path)
    if args.model == 'W2VAASIST':
        model = XLSRAASIST(model_dir=args.SSL_path).cuda()
    if args.model == 'Rawaasist':
        model = Rawaasist().cuda()
    if args.model == 'mellcnn':
        model = MELLCNN().cuda()
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    test_set = NCSSD_mimi(audio_folder_path= args.id_source_mimi)      
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    with open('./result/result_ncssd_mimi.txt', 'w') as f:
        for idx, data_slice in enumerate(tqdm(testDataLoader)):
            waveform, filenames,labels = data_slice[0],data_slice[1],data_slice[2] 
            with torch.no_grad(): 
                feats, logits = model(waveform)
            
            scores = torch.nn.functional.softmax(logits, dim=1)
            vectors, predicted = torch.max(scores, dim=1)
            for i in range(feats.shape[0]):
                filename = filenames[i]  
                pred_label = int(predicted[i].item())  
                score = vectors[i].item()  
                f.write(f'{filename} {pred_label} {score}\n') 
                

    test_set = NCSSD_snac(audio_folder_path= args.id_source_snac)                     
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    with open('./result/result_ncssd_snac.txt', 'w') as f:
        for idx, data_slice in enumerate(tqdm(testDataLoader)):
            waveform, filenames,labels = data_slice[0],data_slice[1],data_slice[2] 
            with torch.no_grad(): 
                feats, logits = model(waveform)
            scores = torch.nn.functional.softmax(logits, dim=1)
            vectors, predicted = torch.max(scores, dim=1)
            for i in range(feats.shape[0]):
                filename = filenames[i]  
                pred_label = int(predicted[i].item())  
                score = vectors[i].item()  
                f.write(f'{filename} {pred_label} {score}\n')                 
                
                
def source_tracing_id_config(feat_model_path):
    checkpoint = torch.load(feat_model_path)
    if args.model == 'W2VAASIST':
        model = XLSRAASIST(model_dir=args.SSL_path).cuda()
    if args.model == 'Rawaasist':
        model = Rawaasist().cuda()
    if args.model == 'mellcnn':
        model = MELLCNN().cuda()
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    folders = ["C3-1", "C3-2", "C3-3", "C3-4", "C4-1", "C4-2", "C4-3"]
    
    for folder in folders:
        audio_folder_path = os.path.join(args.id_config, folder)
        test_set = id_diffconfig(audio_folder_path=audio_folder_path)
        testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
        result_file_path = f'./result/result_{folder}.txt'
        
        with open(result_file_path, 'w') as f:
            for idx, data_slice in enumerate(tqdm(testDataLoader)):
                waveform, filenames, labels = data_slice[0], data_slice[1], data_slice[2]  

                with torch.no_grad(): 
                    feats, logits = model(waveform)
                
                scores = torch.nn.functional.softmax(logits, dim=1)
                vectors, predicted = torch.max(scores, dim=1)
                
                for i in range(feats.shape[0]):
                    filename = filenames[i]  
                    pred_label = int(predicted[i].item())  
                    score = vectors[i].item()  
                    f.write(f'{filename} {pred_label} {score}\n') 
                
                                
def source_tracing_id_unseenreal(feat_model_path):
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
    
    # Define your test sets
    test_sets = {
        "ASVspoof2019LA_real": ASVspoof2019LA_real(path_to_features = args.id_unseenreal_19_wav,path_to_protocol= args.id_unseenreal_19_label),
        "NCSSD": NCSSD(audio_folder_path = args.id_unseenreal_ncssd),
        "ITW_real": ITW_real(path_to_audio = args.id_unseenreal_itw, path_to_protocol= args.id_unseenreal_itw_label)
    }

    # Define your output files
    output_files = {
        "ASVspoof2019LA_real": './result/result_unseenreal_19.txt',
        "NCSSD": './result/result_unseenreal_ncssd.txt',
        "ITW_real": './result/result_unseenreal_itw.txt'
    }

    # Process each test set
    for key in test_sets:
        test_set = test_sets[key]
        output_file = output_files[key]
        
        testDataLoader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)

        with open(output_file, 'w') as f:
            for idx, data_slice in enumerate(tqdm(testDataLoader)):
                waveform, filenames, labels = data_slice[0], data_slice[1], data_slice[2]  
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
    
    source_tracing_id_source(model_path)
    source_tracing_id_config(model_path)
    source_tracing_id_unseenreal(model_path)