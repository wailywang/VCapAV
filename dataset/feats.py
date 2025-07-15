import torch, torch.nn as nn, random
from torchaudio import transforms
# import sandbox.util_frontend as front_end

# class LFCC_Cal(nn.Module):
#     ### The `n_mels` here denotes the dim of LFCC, should be 20 by default.
#     def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, **kargs): 
#         super(LFCC_Cal, self).__init__()
#         self.LFCC_extractor = front_end.LFCC(fl = win_length,
#                                 fs=hop_length,
#                                 fn=n_fft,
#                                 sr=sample_rate,
#                                 filter_num=20,
#                                 with_energy=True,
#                                 with_delta=True,
#                                 max_freq = 0.5)
    
#     def forward(self, x, is_aug=[]):
#         out = self.LFCC_extractor(x)
#         out = out.transpose(1,2)
#         return out

class logFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, window_fn='hann', n_mels=80, trim=None, **kwargs):
        super().__init__()
        if window_fn == 'hamming':
            window_fn = torch.hamming_window
        elif window_fn == 'hann':
            window_fn = torch.hann_window
        elif window_fn == 'blackman':
            window_fn = torch.blackman_window
        else:
            raise ValueError('window_fn should be one of "hamming", "hann", "blackman"')
        self.fbankCal = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=n_fft,
                                                  win_length=win_length,
                                                  hop_length=hop_length,
                                                  window_fn=window_fn,
                                                  n_mels=n_mels)
        self.trim = trim

    def forward(self, input_signal, is_aug=[], **kwargs):
        x = input_signal
        if self.trim:
            out = self.fbankCal(x)[:,:self.trim,:]
        else:
            out = self.fbankCal(x)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        length = torch.LongTensor([out.shape[-1]]*out.shape[0])
        
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
        return out, length

class logSpecCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, window_fn, **kwargs):
        super(logSpecCal, self).__init__()
        if window_fn == 'hamming':
            window_fn = torch.hamming_window
        elif window_fn == 'hann':
            window_fn = torch.hann_window
        elif window_fn == 'blackman':
            window_fn = torch.blackman_window
        else:
            raise ValueError('window_fn should be one of "hamming", "hann", "blackman"')
        
        self.specCal = transforms.Spectrogram(n_fft=n_fft,
                                            win_length=win_length,
                                            hop_length=hop_length,
                                            window_fn=window_fn) # (channel, n_ffts, time)

    def forward(self, input_signal, is_aug=[], **kwargs):
        out = self.specCal(input_signal)
        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        length = torch.LongTensor([out.shape[-1]]*out.shape[0])
        # for i in range(len(is_aug)):
        #     if is_aug[i]:
        #         rn = out[i].mean()
        #         for n in range(random.randint(2, 5)):
        #             offset = random.randint(5, 6)
        #             start = random.randrange(0, out.shape[1] - offset)
        #             out[i][start : start+offset] = rn     
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
        return out, length


class logSpecCal_low(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, window_fn='hann', **kwargs):
        super(logSpecCal_low, self).__init__()
        if window_fn == 'hamming':
            window_fn = torch.hamming_window
        elif window_fn == 'hann':
            window_fn = torch.hann_window
        elif window_fn == 'blackman':
            window_fn = torch.blackman_window
        else:
            raise ValueError('window_fn should be one of "hamming", "hann", "blackman"')
        
        self.specCal = transforms.Spectrogram(n_fft=n_fft,
                                            win_length=win_length,
                                            hop_length=hop_length,
                                            window_fn=window_fn) # (channel, n_ffts, time)

    def forward(self, input_signal, is_aug=[], **kwargs):
        out = self.specCal(input_signal)
        out = torch.log(out + 1e-6)
        out = out[:,:out.shape[1]//2,:]
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        length = torch.LongTensor([out.shape[-1]]*out.shape[0])
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
        return out, length



class Raw_Cal(nn.Module):
    ### Return: x ###
    def __init__(self, **kargs) -> None:
        super().__init__()
    def forward(self, input_signal, is_aug=[], **kwargs):
        out = input_signal.contiguous()
        length = torch.LongTensor([out.shape[-1]]*out.shape[0])
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
        return out, length

if __name__ == '__main__':
    x = torch.zeros(2,112000)
    # model = logFbankCal(sample_rate=16000, n_fft=1024, win_length=int(0.064*16000), hop_length=int(0.008*16000), n_mels=80)
    model = logFbankCal(sample_rate=16000, n_fft=512, win_length=int(0.025*16000), hop_length=int(0.01*16000), n_mels=80)
    # model = logFbankCal(sample_rate=16000, n_fft=1024, win_length=400, hop_length=128, n_mels=80)
    model.eval()
    out, length = model(x)
    print(out.shape, length) # (B, F, T), (T, T)