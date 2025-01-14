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



def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 160000
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform


def normalization(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    distance = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(distance)
    return norm_data



def source_tracing_ood7(feat_model_path):
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

    test_set = codecfake_st_eval(path_to_features=args.ood_wav,
                                path_to_protocol=args.ood_label_7)
                                 
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    with open('./result/result_ood7.txt', 'w') as f:
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

    torch.save({"feas": evalfeas, "logits": evallogits, "labels": evallabels}, './save_feats/evaldict_ood7.pt')


def source_tracing_train(feat_model_path):
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

    test_set = codecfake_st_eval(path_to_features= args.train_wav,
                                path_to_protocol= args.train_label)
                                 
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)


    for idx, data_slice in enumerate(tqdm(testDataLoader)):
        waveform, filenames,labels = data_slice[0],data_slice[1],data_slice[2] # torch.Size([1, 97, 768]
        evallabels.append(torch.tensor(labels).cpu())      
        with torch.no_grad(): 
            feats, logits = model(waveform)
        
        scores = torch.nn.functional.softmax(logits, dim=1)

        evallogits.append(logits)
        evalfeas.append(feats)

    evallabels = torch.cat(evallabels, dim=0)
    evalfeas = torch.cat(evalfeas, dim=0)
    evallogits = torch.cat(evallogits, dim=0)

    torch.save({"feas": evalfeas, "logits": evallogits, "labels": evallabels}, './save_feats/traindict.pt')


def source_tracing_ood8_11(feat_model_path):
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

    test_set = codecfake_st_eval(path_to_features=args.ood_wav,
                                path_to_protocol= args.ood_label_8_11)
                                 
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    with open('./result/result_ood8.txt', 'w') as f:
        for idx, data_slice in enumerate(tqdm(testDataLoader)):
            waveform, filenames,labels = data_slice[0],data_slice[1],data_slice[2] # torch.Size([1, 97, 768]
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

    torch.save({"feas": evalfeas, "logits": evallogits, "labels": evallabels}, './save_feats/evaldict_ood8.pt')


def source_tracing_oodvoc(feat_model_path):
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

    evaldataset_ood1 = ITW_fake(path_to_audio = args.id_unseenreal_itw, path_to_protocol= args.id_unseenreal_itw_label)
    evaldataset_ood2 = ASVspoof2019LA_fake(path_to_audio = args.id_unseenreal_19_wav,path_to_protocol= args.id_unseenreal_19_label)
    test_set = [evaldataset_ood1,evaldataset_ood2]
    test_set = ConcatDataset(test_set)
                                 
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    with open('./result/result_oodvoc.txt', 'w') as f:
        for idx, data_slice in enumerate(tqdm(testDataLoader)):
            waveform, filenames,labels = data_slice[0],data_slice[1],data_slice[2] # torch.Size([1, 97, 768]
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


    evallabels = torch.cat(evallabels, dim=0)
    evalfeas = torch.cat(evalfeas, dim=0)
    evallogits = torch.cat(evallogits, dim=0)

    torch.save({"feas": evalfeas, "logits": evallogits, "labels": evallabels}, './save_feats/evaldict_oodvoc.pt')


def source_tracing_ood_config(feat_model_path):

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

    test_set = ood_diff(audio_folder_path = args.ood_diff_config)
                                 
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    for idx, data_slice in enumerate(tqdm(testDataLoader)):
        waveform, filenames,labels = data_slice[0],data_slice[1],data_slice[2] 
        evallabels.append(torch.tensor(labels).cpu())      
        with torch.no_grad(): 
            feats, logits = model(waveform)
        
        scores = torch.nn.functional.softmax(logits, dim=1)
        evallogits.append(logits)
        evalfeas.append(feats)

    evallabels = torch.cat(evallabels, dim=0)
    evalfeas = torch.cat(evalfeas, dim=0)
    evallogits = torch.cat(evallogits, dim=0)

    torch.save({"feas": evalfeas, "logits": evallogits, "labels": evallabels}, './save_feats/evaldict_ood_config.pt')

def source_tracing_ood_source(feat_model_path):

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

    test_set = ood_diff(audio_folder_path = args.ood_diff_source)
                                 
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    for idx, data_slice in enumerate(tqdm(testDataLoader)):
        waveform, filenames,labels = data_slice[0],data_slice[1],data_slice[2] # torch.Size([1, 97, 768]
        evallabels.append(torch.tensor(labels).cpu())      
        with torch.no_grad(): 
            feats, logits = model(waveform)
        
        scores = torch.nn.functional.softmax(logits, dim=1)
        evallogits.append(logits)
        evalfeas.append(feats)

    evallabels = torch.cat(evallabels, dim=0)
    evalfeas = torch.cat(evalfeas, dim=0)
    evallogits = torch.cat(evallogits, dim=0)

    torch.save({"feas": evalfeas, "logits": evallogits, "labels": evallabels}, './save_feats/evaldict_ood_source.pt')





if __name__ == "__main__":
    args = init()


    model_dir = os.path.join(args.model_folder)

    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")

    source_tracing_train(model_path)      # save training feature, logits, label, prepare to OOD method
    source_tracing_ood7(model_path)
    source_tracing_ood8_11(model_path)
    source_tracing_oodvoc(model_path)
    source_tracing_ood_config(model_path)
    source_tracing_ood_source(model_path)
    
