import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, Sampler
import torch.utils.data.sampler as torch_sampler
from collections import defaultdict
from tqdm import tqdm, trange
import random
from utils import *
import eval_metrics as em
from feature_extraction import *
from rawaasist import *
torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)
import config

def initParams():

    args = config.initParams()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))



        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def shuffle(feat,  labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    # this_len = this_len[shuffle_index]
    return feat, labels


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'W2VAASIST':
        feat_model = XLSRAASIST(model_dir=args.SSL_path).cuda()

    if args.model == 'Rawaasist':
        feat_model = Rawaasist().cuda()

    if args.model == 'mellcnn':
        feat_model = MELLCNN().cuda()
        
        
        
    if args.continue_training:
        feat_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')).to(args.device)
    #feat_model = nn.DataParallel(feat_model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    
    codecfake_st_trainingset = codecfake_st(args.train_wav,args.train_label)
    codecfake_st_devset = codecfake_st(args.dev_wav,args.dev_label)


    training_set = codecfake_st_trainingset
    validation_set = codecfake_st_devset
    # for dataset in datasets:
    #     print(len(dataset),f"Dataset {dataset} length")
    #     assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."


    # training_set = ConcatDataset(datasets)
    trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size),
                            shuffle=False, num_workers=args.num_workers,
                            sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))))                     
    valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size),
                                    shuffle=False, num_workers=args.num_workers,
                                    sampler=torch_sampler.SubsetRandomSampler(range(len(validation_set))))


    trainOri_flow = iter(trainOriDataLoader)
    valOri_flow = iter(valOriDataLoader)


    # weight = torch.FloatTensor([10,1]).to(args.device)   # concentrate on real 0

    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss()

    else:
        criterion = nn.functional.binary_cross_entropy()

    prev_loss = 1e8
    prev_eer = 1
    monitor_loss = 'base_loss'
  
    for epoch_num in tqdm(range(args.num_epochs)):

        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        testlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                feat, audio_fn,  labels = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                feat, audio_fn,  labels = next(trainOri_flow)
            labels = labels.to(args.device) 


            feat_optimizer.zero_grad()
            feats, feat_outputs = feat_model(feat)
            feat_loss = criterion(feat_outputs, labels)
            feat_loss.backward()
            feat_optimizer.step()


            trainlossDict['base_loss'].append(feat_loss.item())

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                            str(trainlossDict[monitor_loss][-1]) + "\n")

        feat_model.eval()
        with torch.no_grad():
            ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    feat, audio_fn, labels= next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    feat, audio_fn, labels= next(valOri_flow)
                    
                labels = labels.to(args.device) 

                feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs[:, 0]
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]

                ip1_loader.append(feats)
                idx_loader.append((labels))
                devlossDict["base_loss"].append(feat_loss.item())
                score_loader.append(score)

                desc_str = ''
                for key in sorted(devlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '
                # v.set_description(desc_str)
                print(desc_str)
            valLoss = np.nanmean(devlossDict[monitor_loss])
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()

            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]   # only binary eer
            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(np.nanmean(devlossDict[monitor_loss])) + "\t" + str(val_eer) +"\n")
            print("Val EER: {}".format(val_eer))

        if (epoch_num + 1) % 1 == 0:
            torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))

        if valLoss < prev_loss:
            # Save the model checkpoint
            torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            prev_loss = valLoss


    return feat_model, valLoss


if __name__ == "__main__":
    args = initParams()
    _, _ = train(args)
