import sys, os, math, random, warnings, torch, librosa, numpy as np, torchaudio
from scipy.signal import fftconvolve
from python_speech_features import sigproc
from torch.utils.data import Dataset
# 获取当前文件的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录的路径
parent_dir = os.path.dirname(current_dir)
# 将上层目录添加到sys.path中
sys.path.append(parent_dir)

from ddataset.sampler import WavBatchSampler
from utils.RawBoost import process_Rawboost_feature
from utils.FRAM_RIR import FRAM_RIR, single_channel

class TrainDevDataset(Dataset):
    def __init__(self, args, wav_scp, utt2label:dict,
                 fs=16000, preemph=False,
                 is_aug=False, aug_rate:float=2/3, snr_range=None, noise_dict=None,
                 is_specaug=False, vad=False, speed=False, reverb=False):
        # argsparse
        self.args=args
        # data info
        self.wav_scp = wav_scp
        self.utt2label = utt2label
        self.num_utt = len(self.wav_scp)
        self.fs = fs 
        # audio process
        self.preemph = preemph
        self.vad = vad
        # augmentation
        self.is_aug = is_aug
        self.aug_rate = aug_rate
        self.is_specaug = is_specaug # SpecAug
        self.noise_dict = noise_dict # noise type
        self.snr = snr_range # noise snr range
        self.speed = speed # tempo re-speed
        self.reverb = reverb # online reverberation

    def __len__(self):
        return self.num_utt

    def _load_data(self, file_path):
        signal, fs = librosa.load(file_path, sr=self.fs)
        return signal
    
    def _norm_speech(self, signal):
        if np.abs(signal).max() == 0:
            return signal
        signal = signal / (np.abs(signal).max())
        return signal
    
    def _augmentation(self, signal, file_path, speed=False):        
        noise_types = ['noise']#, reverb]
        if self.is_specaug:
            noise_types += ['spec_aug']
        if speed == True:
            noise_types += ['sox']
        noise_types = random.choice(noise_types)
        
        if noise_types == 'spec_aug':
            return signal, 1 # indicator to apply specAug at feature calculator
        
        elif noise_types == 'sox':
            # noise_types = random.choice(['tempo', 'vol'])
            noise_types = 'tempo'
            if noise_types == 'tempo':
                val = random.choice([0.9, 1.1])
                effect = [['tempo', str(val)]]
            else:
                val = random.random() * 15 + 5
                effect = [['vol', str(val)]]
                
            signal_sox, _ = torchaudio.sox_effects.apply_effects_tensor(torch.tensor(signal.astype('float32').reshape(1, -1)), self.fs, effect)# using CPU
            return self._truncate_speech(signal_sox.numpy()[0], len(signal)), 0

        # elif noise_types == 'reverb':
        #     power = (signal ** 2).mean()
        #     rir = self._load_data(random.choice(self.noise_dict[noise_types]))
        #     signal = fftconvolve(rir, signal)[:signal.shape[0]]
        #     power2 = (signal ** 2).mean()
        #     signal = np.sqrt(power / max(power2, 1e-10)) * signal
        #     return signal, 0
        
        else:
            noise_signal = np.zeros(signal.shape[0], dtype='float32')
            for noise_type in random.choice([['noise'], ['music'], ['babb', 'music'], ['babb'] * random.randint(3, 8)]):
                noise = self._load_data(random.choice(self.noise_dict[noise_type]))
                noise = self._truncate_speech(noise, signal.shape[0])
                noise_signal = noise_signal + self._norm_speech(noise)
            snr = random.uniform(self.snr[0], self.snr[1])
            power = (signal ** 2).mean()
            noise_power = (noise_signal ** 2).mean()
            sigma_n = (
                10 ** (-snr / 20)
                * np.sqrt(power)
                / np.sqrt(max(noise_power, 1e-10))
            )
            return signal + noise_signal * sigma_n, 0
    def _reverb(self, signal):
            simu_config = {
                "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
                "rt60": [0.2, 1.0],
                "sr": 16000,
                "mic_dist": [0.2, 5.0],
                "num_src": 1}
            rir, rir_direct = single_channel(simu_config)
            power = (signal ** 2).mean()

            late_reverb_signal = fftconvolve(rir, signal)[:signal.shape[0]]
            power2 = (late_reverb_signal ** 2).mean()
            late_reverb_signal = np.sqrt(power / max(power2, 1e-10)) * late_reverb_signal

            direct_signal = fftconvolve(rir_direct, signal)[:signal.shape[0]]
            power2 = (direct_signal ** 2).mean()
            direct_signal = np.sqrt(power / max(power2, 1e-10)) * direct_signal

            return direct_signal, late_reverb_signal

    def _truncate_speech(self, signal, tlen, offset=0):
        if tlen == None:
            return signal
        if signal.shape[0] <= tlen:
            signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
        if offset:
            offset = random.randint(0, signal.shape[0] - tlen)
        return np.array(signal[offset : offset + tlen])


    def __getitem__(self, idx):
        if isinstance(idx, int):
            tlen = None
        elif len(idx) == 2:
            idx, tlen = idx
            tlen = int(tlen * self.fs)
        else:
            raise AssertionError("The idx should be int or a list with length of 2.")

        utt, file_path = self.wav_scp[idx]
        signal = self._load_data(file_path)

        try:
            label = self.utt2label[utt]
        except KeyError:
            print(f"KeyError: {utt} not found in utt2label")
            raise

        if self.vad:
            signal, _ = librosa.effects.trim(signal, top_db=40)

        signal = self._truncate_speech(signal, tlen)  # repeat and cut

        signal = process_Rawboost_feature(signal, self.fs, self.args, self.args.algo)  # algo==0 means no RawBoost processing

        signal = self._norm_speech(signal)
        is_spec_aug = 0

        if self.is_aug and random.random() < self.aug_rate:
            aug_signal, is_spec_aug = self._augmentation(signal, file_path, self.speed)
            aug_signal = self._norm_speech(aug_signal)
        else:
            aug_signal = signal

        if self.reverb and random.random() < self.aug_rate:
            signal, aug_signal = self._reverb(aug_signal)
        else:
            signal, aug_signal = aug_signal, aug_signal

        if self.preemph:
            signal = sigproc.preemphasis(signal, 0.97)
            aug_signal = sigproc.preemphasis(aug_signal, 0.97)

        signal = torch.from_numpy(signal.astype('float32'))
        aug_signal = torch.from_numpy(aug_signal.astype('float32'))

        return signal, aug_signal, is_spec_aug, label, utt


class TrainDevDataset_ASVspoof5(Dataset):
    def __init__(self, args, wav_scp, utt2label:dict,
                 fs=16000, preemph=False,
                 is_aug=False, aug_rate:float=2/3, snr_range=None, noise_dict=None,
                 is_specaug=False, vad=False, speed=False, reverb=False):
        # argsparse
        self.args=args
        # data info
        self.wav_scp = wav_scp
        self.utt2label = utt2label
        self.num_utt = len(self.wav_scp)
        self.fs = fs 
        # audio process
        self.preemph = preemph
        self.vad = vad
        # augmentation
        self.is_aug = is_aug
        self.aug_rate = aug_rate
        self.is_specaug = is_specaug # SpecAug
        self.noise_dict = noise_dict # noise type
        self.snr = snr_range # noise snr range
        self.speed = speed # tempo re-speed
        self.reverb = reverb # online reverberation

    def __len__(self):
        return self.num_utt

    def _load_data(self, file_path):
        signal, fs = librosa.load(file_path, sr=self.fs)
        return signal
    
    def _norm_speech(self, signal):
        if np.abs(signal).max() == 0:
            return signal
        signal = signal / (np.abs(signal).max())
        return signal
    
    def _augmentation(self, signal, file_path, speed=False):        
        noise_signal = np.zeros(signal.shape[0], dtype='float32')
        for noise_type in ['noise']:
            noise = self._load_data(random.choice(self.noise_dict[noise_type]))
            noise = self._truncate_speech(noise, signal.shape[0])
            noise_signal = noise_signal + self._norm_speech(noise)
        snr = random.uniform(self.snr[0], self.snr[1])
        power = (signal ** 2).mean()
        noise_power = (noise_signal ** 2).mean()
        sigma_n = (
            10 ** (-snr / 20)
            * np.sqrt(power)
            / np.sqrt(max(noise_power, 1e-10))
        )
        return signal + noise_signal * sigma_n, 0
        
    def _reverb(self, signal):
            simu_config = {
                "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
                "rt60": [0.2, 1.0],
                "sr": 16000,
                "mic_dist": [0.2, 5.0],
                "num_src": 1}
            rir, rir_direct = single_channel(simu_config)
            power = (signal ** 2).mean()

            late_reverb_signal = fftconvolve(rir, signal)[:signal.shape[0]]
            power2 = (late_reverb_signal ** 2).mean()
            late_reverb_signal = np.sqrt(power / max(power2, 1e-10)) * late_reverb_signal

            direct_signal = fftconvolve(rir_direct, signal)[:signal.shape[0]]
            power2 = (direct_signal ** 2).mean()
            direct_signal = np.sqrt(power / max(power2, 1e-10)) * direct_signal

            return direct_signal, late_reverb_signal

    def _truncate_speech(self, signal, tlen, offset=False):
        if tlen == None:
            return signal
        if signal.shape[0] <= tlen:
            signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
        if offset:
            offset = random.randint(0, signal.shape[0] - tlen)
        return np.array(signal[offset : offset + tlen])


    def __getitem__(self, idx):
        if isinstance(idx, int):
            tlen = None
        elif len(idx) == 2:
            idx, tlen = idx
            tlen = int(tlen * self.fs)
        else:
            raise AssertionError("The idx should be int or a list with length of 2.")

        utt, file_path = self.wav_scp[idx]
        signal = self._load_data(file_path)
        label = self.utt2label[utt]

        if self.vad:
            signal, _ = librosa.effects.trim(signal,top_db=40)

        signal = self._truncate_speech(signal, tlen, self.args.offset) # repeat and cut

        signal = process_Rawboost_feature(signal, self.fs, self.args, self.args.algo) # algo==0 means no RawBoost processing

        signal = self._norm_speech(signal)
        is_spec_aug = 0

        if self.is_aug and random.random() < self.aug_rate:
            aug_signal, is_spec_aug = self._augmentation(signal, file_path, self.speed)
            aug_signal = self._norm_speech(aug_signal)
        else:
            aug_signal = signal
        
        if self.reverb and random.random() < self.aug_rate:
            signal, aug_signal = self._reverb(aug_signal)
        else:
            signal, aug_signal = aug_signal, aug_signal

        if self.preemph:
            signal = sigproc.preemphasis(signal, 0.97)
            aug_signal = sigproc.preemphasis(aug_signal, 0.97)
        
        signal = torch.from_numpy(signal.astype('float32'))
        aug_signal = torch.from_numpy(aug_signal.astype('float32'))
        
        return signal, aug_signal, is_spec_aug, label, utt

class TrainDevDataset_ASVspoof5_offset(Dataset):
    def __init__(self, args, wav_scp, utt2label:dict,
                 fs=16000, preemph=False,
                 is_aug=False, aug_rate:float=2/3, snr_range=None, noise_dict=None,
                 is_specaug=False, vad=False, speed=False, reverb=False):
        # argsparse
        self.args=args
        # data info
        self.wav_scp = wav_scp
        self.utt2label = utt2label
        self.num_utt = len(self.wav_scp)
        self.fs = fs 
        # audio process
        self.preemph = preemph
        self.vad = vad
        # augmentation
        self.is_aug = is_aug
        self.aug_rate = aug_rate
        self.is_specaug = is_specaug # SpecAug
        self.noise_dict = noise_dict # noise type
        self.snr = snr_range # noise snr range
        self.speed = speed # tempo re-speed
        self.reverb = reverb # online reverberation

    def __len__(self):
        return self.num_utt

    def _load_data(self, file_path):
        signal, fs = librosa.load(file_path, sr=self.fs)
        return signal
    
    def _norm_speech(self, signal):
        if np.abs(signal).max() == 0:
            return signal
        signal = signal / (np.abs(signal).max())
        return signal
    
    def _augmentation(self, signal, file_path, speed=False):        
        noise_signal = np.zeros(signal.shape[0], dtype='float32')
        for noise_type in ['noise']:
            noise = self._load_data(random.choice(self.noise_dict[noise_type]))
            noise = self._truncate_speech(noise, signal.shape[0])
            noise_signal = noise_signal + self._norm_speech(noise)
        snr = random.uniform(self.snr[0], self.snr[1])
        power = (signal ** 2).mean()
        noise_power = (noise_signal ** 2).mean()
        sigma_n = (
            10 ** (-snr / 20)
            * np.sqrt(power)
            / np.sqrt(max(noise_power, 1e-10))
        )
        return signal + noise_signal * sigma_n, 0
        
    def _reverb(self, signal):
            simu_config = {
                "min_max_room": [[3, 3, 2.5], [10, 6, 4]],
                "rt60": [0.2, 1.0],
                "sr": 16000,
                "mic_dist": [0.2, 5.0],
                "num_src": 1}
            rir, rir_direct = single_channel(simu_config)
            power = (signal ** 2).mean()

            late_reverb_signal = fftconvolve(rir, signal)[:signal.shape[0]]
            power2 = (late_reverb_signal ** 2).mean()
            late_reverb_signal = np.sqrt(power / max(power2, 1e-10)) * late_reverb_signal

            direct_signal = fftconvolve(rir_direct, signal)[:signal.shape[0]]
            power2 = (direct_signal ** 2).mean()
            direct_signal = np.sqrt(power / max(power2, 1e-10)) * direct_signal

            return direct_signal, late_reverb_signal

    def _truncate_speech(self, signal, maxlen, offset=False): # e.g. len=7s == maxlen=112000
        x_len = signal.shape[0]
        if maxlen == None:
            return signal
        # if duration is longer than maxlen, truncate it
        if x_len > maxlen and offset:
            offset = np.random.randint(x_len - maxlen)
            return np.array(signal[offset:offset+maxlen])
        elif x_len > maxlen and not offset:
            return np.array(signal[:maxlen])
        # if duration is shorter than maxlen, repeat it
        num_repeats = int(maxlen / x_len) + 1
        return np.tile(signal, num_repeats)[:maxlen]



    def __getitem__(self, idx):
        if isinstance(idx, int):
            tlen = None
        elif len(idx) == 2:
            idx, tlen = idx
            tlen = int(tlen * self.fs)
        else:
            raise AssertionError("The idx should be int or a list with length of 2.")

        utt, file_path = self.wav_scp[idx]
        signal = self._load_data(file_path)
        label = self.utt2label[utt]

        if self.vad:
            signal, _ = librosa.effects.trim(signal,top_db=40)

        signal = self._truncate_speech(signal, tlen, self.args.offset) # repeat and cut

        signal = process_Rawboost_feature(signal, self.fs, self.args, self.args.algo) # algo==0 means no RawBoost processing

        signal = self._norm_speech(signal)
        is_spec_aug = 0

        if self.is_aug and random.random() < self.aug_rate:
            aug_signal, is_spec_aug = self._augmentation(signal, file_path, self.speed)
            aug_signal = self._norm_speech(aug_signal)
        else:
            aug_signal = signal
        
        if self.reverb and random.random() < self.aug_rate:
            signal, aug_signal = self._reverb(aug_signal)
        else:
            signal, aug_signal = aug_signal, aug_signal

        if self.preemph:
            signal = sigproc.preemphasis(signal, 0.97)
            aug_signal = sigproc.preemphasis(aug_signal, 0.97)
        
        signal = torch.from_numpy(signal.astype('float32'))
        aug_signal = torch.from_numpy(aug_signal.astype('float32'))
        
        return signal, aug_signal, is_spec_aug, label, utt

class EvalDataset(Dataset):
    def __init__(self, wav_scp, frame_level=False,
                 fs=16000, preemph=False, vad=False):
        # data info
        self.wav_scp = wav_scp
        self.num_utt = len(self.wav_scp)
        self.fs = fs 
        # audio process
        self.preemph = preemph
        self.vad = vad
        # frame level
        self.frame_level = frame_level

    def __len__(self):
        return self.num_utt

    def _load_data(self, file_path):
        signal, fs = librosa.load(file_path, sr=self.fs)
        return signal
    
    def _norm_speech(self, signal):
        if np.abs(signal).max() == 0:
            return signal
        signal = signal / (np.abs(signal).max())
        return signal

    def _truncate_speech(self, signal, tlen, offset=0):
        if tlen == None:
            return signal
        if signal.shape[0] <= tlen:
            signal = np.concatenate([signal] * (tlen // signal.shape[0] + 1), axis=0)
        if offset is None:
            offset = random.randint(0, signal.shape[0] - tlen)
        return np.array(signal[offset : offset + tlen])


    def __getitem__(self, idx):
        if isinstance(idx, int):
            tlen = None
        elif len(idx) == 2:
            idx, tlen = idx
            tlen = int(tlen * self.fs)
        else:
            raise AssertionError("The idx should be int or a list with length of 2.")

        utt, file_path = self.wav_scp[idx]

        signal = self._load_data(file_path)
        
        if self.vad:
            signal, _ = librosa.effects.trim(signal,top_db=40)

        if self.frame_level:
            _sig_list = []
            if signal.shape[0] <= tlen:
                _sig_list.append(self._truncate_speech(signal, tlen))
            else:
                for i in range(math.floor(signal.shape[0] / tlen)):
                    _sig_list.append(self._truncate_speech(signal, tlen, offset=i*tlen))
            
            sig_list = []
            for signal in _sig_list:
                signal = self._norm_speech(signal)
                if self.preemph:
                    signal = sigproc.preemphasis(signal, 0.97)
                signal = torch.from_numpy(signal.astype('float32'))
                sig_list.append(signal)
            utt_list = [utt] * len(sig_list)
            return sig_list, utt_list
        else:
            signal = self._truncate_speech(signal, tlen)
            signal = self._norm_speech(signal)
            if self.preemph:
                signal = sigproc.preemphasis(signal, 0.97)
            signal = torch.from_numpy(signal.astype('float32'))
            return signal, utt


class EvalDataset_offset(Dataset):
    def __init__(self, wav_scp, frame_level=False,
                 fs=16000, preemph=False, vad=False):
        # data info
        self.wav_scp = wav_scp
        self.num_utt = len(self.wav_scp)
        self.fs = fs 
        # audio process
        self.preemph = preemph
        self.vad = vad
        # frame level
        self.frame_level = frame_level

    def __len__(self):
        return self.num_utt

    def _load_data(self, file_path):
        signal, fs = librosa.load(file_path, sr=self.fs)
        return signal
    
    def _norm_speech(self, signal):
        if np.abs(signal).max() == 0:
            return signal
        signal = signal / (np.abs(signal).max())
        return signal

    def _truncate_speech(self, signal, maxlen, offset=False): # e.g. len=7s == maxlen=112000
        x_len = signal.shape[0]
        if maxlen == None:
            return signal
        # if duration is longer than maxlen, truncate it
        if x_len > maxlen and offset:
            offset = np.random.randint(x_len - maxlen)
            return np.array(signal[offset:offset+maxlen])
        elif x_len > maxlen and not offset:
            return np.array(signal[:maxlen])
        # if duration is shorter than maxlen, repeat it
        num_repeats = int(maxlen / x_len) + 1
        return np.tile(signal, num_repeats)[:maxlen]


    def __getitem__(self, idx):
        if isinstance(idx, int):
            tlen = None
        elif len(idx) == 2:
            idx, tlen = idx
            tlen = int(tlen * self.fs)
        else:
            raise AssertionError("The idx should be int or a list with length of 2.")

        utt, file_path = self.wav_scp[idx]

        signal = self._load_data(file_path)
        
        if self.vad:
            signal, _ = librosa.effects.trim(signal,top_db=40)

        
        signal = self._truncate_speech(signal, tlen, offset=True) # repeat and cut
        signal = self._norm_speech(signal)
        if self.preemph:
            signal = sigproc.preemphasis(signal, 0.97)
        signal = torch.from_numpy(signal.astype('float32'))
        return signal, utt

if __name__ == "__main__":
    key2int = {'bonafide':1,'spoof':0,'genuine':1,'fake':0}
    ## training data
    utt2wav = [line.split() for line in open(f'/SMIIPdata2/ASVspoof5/data/wangyx/scp/train/wav.scp')]
    utt2label = [line.split() for line in open(f'/SMIIPdata2/ASVspoof5/data/wangyx/scp/train/utt2label')]
    
    ##########partial data debug##############
    # 将两个列表合并为一个列表
    combined = list(zip(utt2wav, utt2label))
    # 打乱这个列表
    random.shuffle(combined)
    # 取出10个数据
    utt2wav[:], utt2label[:] = zip(*combined[:10])
  
    
    utt2label = {u:key2int[s] for u, s in utt2label}
    # random.shuffle(utt2wav)
    import argparse
    from torch.utils.data import DataLoader
    args = argparse.Namespace()
    args.algo = 0
    args.offset = 1
    args.dur_range = [4, 4]
    args.batch_size = 64
    trn_dataset = TrainDevDataset_ASVspoof5(args, utt2wav, utt2label, fs=16000)
    trn_sampler = WavBatchSampler(trn_dataset, args.dur_range, shuffle=True, batch_size=2, drop_last=True)
    # 创建 DataLoader 对象
    trn_loader = DataLoader(trn_dataset, batch_sampler=trn_sampler, num_workers=1, pin_memory=True)

    # 迭代并打印数据集的内容
    for i, (signal, aug_signal, is_spec_aug, label, utt) in enumerate(trn_loader):
        print(f"Batch {i+1}: {signal.shape}")
        if i >= 9:  # 打印前10个批次的数据
            break