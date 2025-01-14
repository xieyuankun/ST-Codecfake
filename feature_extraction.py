import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import numpy as np
from utils_dsp import LinearDCT
import librosa
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch.nn as nn
import torch
from functools import reduce
from operator import mul

from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
def trimf(x, params):
    """
    trimf: similar to Matlab definition
    https://www.mathworks.com/help/fuzzy/trimf.html?s_tid=srchtitle

    """
    if len(params) != 3:
        print("trimp requires params to be a list of 3 elements")
        sys.exit(1)
    a = params[0]
    b = params[1]
    c = params[2]
    if a > b or b > c:
        print("trimp(x, [a, b, c]) requires a<=b<=c")
        sys.exit(1)
    y = torch.zeros_like(x, dtype=torch.float32)
    if a < b:
        index = np.logical_and(a < x, x < b)
        y[index] = (x[index] - a) / (b - a)
    if b < c:
        index = np.logical_and(b < x, x < c)
        y[index] = (c - x[index]) / (c - b)
    y[x == b] = 1
    return y

def delta(x):
    """ By default
    input
    -----
    x (batch, Length, dim)

    output
    ------
    output (batch, Length, dim)

    Delta is calculated along Length
    """
    length = x.shape[1]
    output = torch.zeros_like(x)
    x_temp = torch_nn_func.pad(x.unsqueeze(1), (0, 0, 1, 1),
                               'replicate').squeeze(1)
    output = -1 * x_temp[:, 0:length] + x_temp[:, 2:]
    return output


class LFCC(torch_nn.Module):
    """ Based on asvspoof.org baseline Matlab code.
    Difference: with_energy is added to set the first dimension as energy

    """

    def __init__(self, fl, fs, fn, sr, filter_num,
                 with_energy=False, with_emphasis=True,
                 with_delta=True):
        super(LFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num

        f = (sr / 2) * torch.linspace(0, 1, fn // 2 + 1)
        filter_bands = torch.linspace(min(f), max(f), filter_num + 2)

        filter_bank = torch.zeros([fn // 2 + 1, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = trimf(
                f, [filter_bands[idx],
                    filter_bands[idx + 1],
                    filter_bands[idx + 2]])
        self.lfcc_fb = torch_nn.Parameter(filter_bank, requires_grad=False)
        self.l_dct = LinearDCT(filter_num, 'dct', norm='ortho')
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        return

    def forward(self, x):
        """

        input:
        ------
         x: tensor(batch, length), where length is waveform length

        output:
        -------
         lfcc_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphasis
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:] - 0.97 * x[:, 0:-1]

        # STFT
        x_stft = torch.stft(x, self.fn, self.fs, self.fl,
                            window=torch.hamming_window(self.fl),
                            onesided=True, pad_mode="constant")
        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()

        # filter bank
        fb_feature = torch.log10(torch.matmul(sp_amp, self.lfcc_fb) +
                                 torch.finfo(torch.float32).eps)

        # DCT
        lfcc = self.l_dct(fb_feature)

        # Add energy
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2) +
                                 torch.finfo(torch.float32).eps)
            lfcc[:, :, 0] = energy

        # Add delta coefficients
        if self.with_delta:
            lfcc_delta = delta(lfcc)
            lfcc_delta_delta = delta(lfcc_delta)
            lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
        else:
            lfcc_output = lfcc

        # done
        return lfcc_output


class STFT(torch_nn.Module):
    """ Short-time Fourier Transform
    Calculate the spectrogram of the raw waveform
    """

    def __init__(self, fl, fs, fn, sr, with_emphasis=True):
        super(STFT, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.with_emphasis = with_emphasis

    def forward(self, x):
        # pre-emphasis
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:] - 0.97 * x[:, 0:-1]
        # STFT
        x_stft = torch.stft(x, self.fn, self.fs, self.fl,
                            window=torch.hamming_window(self.fl),
                            onesided=True, pad_mode="constant")
        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()

        return sp_amp


class Melspec(torch_nn.Module):
    """ Mel-spectrogram
    """
    def __init__(self):
        super(Melspec, self).__init__()

    def forward(self, x):
        melspec = librosa.feature.melspectrogram(y=x[0].numpy(), sr=16000, n_fft=512, hop_length=128)
        return torch.from_numpy(melspec)

class Melspec(torch_nn.Module):
    """ Mel-spectrogram
    """
    def __init__(self):
        super(Melspec, self).__init__()

    def forward(self, x):
        melspec = librosa.feature.melspectrogram(y=x[0].numpy(), sr=16000, n_fft=512, hop_length=128)
        return torch.from_numpy(melspec)
class rolloff(torch_nn.Module):
    """ Mel-spectrogram
    """
    def __init__(self):
        super(rolloff, self).__init__()

    def forward(self, x):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=x[0].numpy(), n_fft= 512, hop_length=128, sr=16000, roll_percent=0.75)
        return torch.from_numpy(spectral_rolloff)




class XLSR(torch.nn.Module):
    def __init__(self, model_dir, device='cuda', sampling_rate=16000):
        super(XLSR, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  

        # Get the hidden states from the Wav2Vec2 model (outputs.hidden_states is a tuple)
        with torch.no_grad():
            output = self.model(feat).hidden_states[5]

        return output

    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output



if __name__ == "__main__":
    wav = torch.randn(32, 64600)

    model_dir = "/data3/xyk/huggingface/wav2vec2-xls-r-300m/"

    model = XLSR(model_dir=model_dir, device='cuda')
    combined_features = model.extract_features(wav)
    print(combined_features.shape,'combined_features')

