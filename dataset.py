#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
from feature_extraction import LFCC
from torch.utils.data.dataloader import default_collate




def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath,sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform

class asvspoof19(Dataset):
    def __init__(self, access_type, path_to_features, path_to_protocol, part='train', feature='LFCC', feat_len=750, pad_chop=True,
                 padding='repeat'):
        super(asvspoof19, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.label = {"spoof": 1, "bonafide": 0}
                       

        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info
    def __len__(self):
        return len(self.all_files)
    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename+'.flac')
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        waveform = waveform.squeeze(dim=0)
        
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)



class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True, padding='repeat', genuine_only=False):
        super(ASVspoof2019, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        elif self.access_type == 'PA':
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        else:
            raise ValueError("Access type should be LA or PA!")
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        if self.genuine_only:
            assert self.access_type == "LA"
            if self.part in ["train", "dev"]:
                num_bonafide = {"train": 2580, "dev": 2548}
                self.all_files = self.all_files[:num_bonafide[self.part]]
            else:
                res = []
                for item in self.all_files:
                    if "bonafide" in item:
                        res.append(item)
                self.all_files = res
                assert len(self.all_files) == 7355

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 6
        featureTensor = torch.load(filepath)

        if self.feature == "wav2vec2_largeraw":
            #featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        #featureTensor = featureTensor.squeeze(dim=0)
        filename =  "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)



class asvspoof5(Dataset):
    def __init__(self, access_type, path_to_features, path_to_protocol, part='train', feature='LFCC', feat_len=750, pad_chop=True,
                 padding='repeat'):
        super(asvspoof5, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.label = {"spoof": 1, "bonafide": 0}
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info
    def __len__(self):
        return len(self.all_files)
    def __getitem__(self, idx):
        _,filename,_,_,_,label = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename+'.flac')
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)












class codecfake(Dataset):
    def __init__(self, access_type, path_to_features,path_to_protocol, part='train', feature='LFCC', feat_len=750, pad_chop=True,
                 padding='repeat', genuine_only=False):
        super(codecfake, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.label = {"fake": 1, "real": 0}
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, label,_ = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)




class codecfake_st(Dataset):
    def __init__(self, path_to_features,path_to_protocol, part='train'):
        super(codecfake_st, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)

        self.label = {"0": 0, "1": 1,"2": 2,"3": 3,"4": 4,"5": 5, "6": 6}
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, _, label = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)



class codecfake_st(Dataset):
    def __init__(self, path_to_features,path_to_protocol):
        super(codecfake_st, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol


        self.label = {"0": 0, "1": 1,"2": 2,"3": 3,"4": 4,"5": 5, "6": 6}
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, _, label = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)




class codecfake_st_eval(Dataset):
    def __init__(self, path_to_features,path_to_protocol):
        super(codecfake_st_eval, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol

        self.label = {"0": 0, "1": 1,"2": 2,"3": 3,"4": 4,"5": 5, "6": 6, "7": 7, "8":8, "9":9, "10":10, "11":11}
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, _, label = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)




class ITW_real(Dataset):
    def __init__(self,path_to_audio,path_to_protocol):
        super(ITW_real, self).__init__()
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split(',') for info in f.readlines()]
            self.all_info = audio_info
        self.all_info = [info for info in audio_info if info[2] == 'bona-fide']
        self.label = {"bona-fide": 0,"spoof": 1}

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, _,label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)




class ASVspoof2019LA_real(Dataset):
    def __init__(self,path_to_features,path_to_protocol):
        super(ASVspoof2019LA_real, self).__init__()
        self.path_to_audio = path_to_features
        self.path_to_protocol = path_to_protocol
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
        
        self.all_info = [info for info in audio_info if info[4] == 'bonafide']
        self.label = {"bonafide": 0,"spoof": 1}

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        _ , filename, _, _,label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename +'.flac')
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)




class NCSSD(Dataset):
    def __init__(self, audio_folder_path):
        super(NCSSD, self).__init__()
        self.path_to_audio = audio_folder_path
        self.all_wav_files = [f for f in os.listdir(self.path_to_audio) if f.endswith('.wav')]
        self.label = 0

    def __len__(self):
        return len(self.all_wav_files)

    def __getitem__(self, idx):

        filename = self.all_wav_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        return waveform, filename, self.label

    def collate_fn(self, samples):

        return default_collate(samples)


class NCSSD_mimi(Dataset):
    def __init__(self, audio_folder_path):
        super(NCSSD_mimi, self).__init__()
        self.path_to_audio = audio_folder_path
        self.all_wav_files = [f for f in os.listdir(self.path_to_audio) if f.endswith('.wav')]
        self.label = 1

    def __len__(self):
        return len(self.all_wav_files)

    def __getitem__(self, idx):

        filename = self.all_wav_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        return waveform, filename, self.label

    def collate_fn(self, samples):

        return default_collate(samples)

class NCSSD_snac(Dataset):
    def __init__(self, audio_folder_path):
        super(NCSSD_snac, self).__init__()
        self.path_to_audio = audio_folder_path
        self.all_wav_files = [f for f in os.listdir(self.path_to_audio) if f.endswith('.wav')]
        self.label = 6

    def __len__(self):
        return len(self.all_wav_files)

    def __getitem__(self, idx):

        filename = self.all_wav_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        return waveform, filename, self.label

    def collate_fn(self, samples):

        return default_collate(samples)
    
class id_diffconfig(Dataset):
    def __init__(self, audio_folder_path):
        super(id_diffconfig, self).__init__()
        self.path_to_audio = audio_folder_path
        self.all_wav_files = [f for f in os.listdir(self.path_to_audio) if f.endswith('.wav')]
        self.label = 3

    def __len__(self):
        return len(self.all_wav_files)

    def __getitem__(self, idx):

        filename = self.all_wav_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        return waveform, filename, self.label

    def collate_fn(self, samples):

        return default_collate(samples)
    


    
    
class ITW_fake(Dataset):
    def __init__(self,path_to_audio,path_to_protocol):
        super(ITW_fake, self).__init__()
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split(',') for info in f.readlines()]
            self.all_info = audio_info
        self.all_info = [info for info in audio_info if info[2] == 'spoof']
        self.label = {"bona-fide": 0,"spoof": 7}

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, _,label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)

class ASVspoof2019LA_fake(Dataset):
    def __init__(self,path_to_audio,path_to_protocol):
        super(ASVspoof2019LA_fake, self).__init__()
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_protocol
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
        
        self.all_info = [info for info in audio_info if info[4] == 'spoof']
        self.label = {"bonafide": 0,"spoof": 7}

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        _ , filename, _, _,label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename +'.flac')
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)

class ood_diff(Dataset):
    def __init__(self, audio_folder_path):
        super(ood_diff, self).__init__()
        self.path_to_audio = audio_folder_path
        self.all_wav_files = [f for f in os.listdir(self.path_to_audio) if f.endswith('.wav')]
        self.label = 7

    def __len__(self):
        return len(self.all_wav_files)

    def __getitem__(self, idx):

        filename = self.all_wav_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        return waveform, filename, self.label

    def collate_fn(self, samples):

        return default_collate(samples)



class ood_alm(Dataset):
    def __init__(self, audio_folder_path):
        super(ood_alm, self).__init__()
        self.path_to_audio = audio_folder_path
        self.all_wav_files = [f for f in os.listdir(self.path_to_audio) if f.endswith('.wav')]


    def __len__(self):
        return len(self.all_wav_files)

    def __getitem__(self, idx):

        filename = self.all_wav_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform)
        return waveform, filename

    def collate_fn(self, samples):

        return default_collate(samples)