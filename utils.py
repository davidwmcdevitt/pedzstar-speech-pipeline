import torchaudio
import torch
import random
import torchaudio.transforms as transforms
from torch.utils.data import Dataset
from IPython.display import Audio

class AudioUtil():
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

  @staticmethod
  def normalize(aud):
    sig, sr = aud
    mean = torch.mean(sig)
    std = torch.std(sig)
    epsilon = 1e-8
    normalized_sig = (sig - mean) / (std + epsilon)
    return (normalized_sig, sr)

  @staticmethod
  def add_noise(aud, noise_level):
    sig, sr = aud
    noise = torch.randn(sig.size())
    scaled_noise = noise * noise_level
    noisy_sig = sig + scaled_noise
    noisy_sig = torch.clamp(noisy_sig, min=-1, max=1)
    return (noisy_sig, sr)

  @staticmethod
  def time_shift(aud, shift_limit):
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

  @staticmethod
  def spectro_gram(aud, n_mels=128, n_fft=2048, hop_len=None):
    sig, sr = aud
    top_db = 80
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

  @staticmethod
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec
    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    return aug_spec

  @staticmethod
  def rechannel(aud, new_channel):
    sig, sr = aud
    if (sig.shape[0] == new_channel):
      return aud
    if (new_channel == 1):
      resig = sig[:1, :]
    else:
      resig = torch.cat([sig, sig])

    return ((resig, sr))

class SoundDS(Dataset):
  def __init__(self, df, data_path, args):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 1000
    self.sr = 44100
    self.channel = 2
    self.shift_pct = 0.4
    self.args = args
    
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    audio_file = self.data_path + self.df.loc[idx, 'relative_path']
    class_id = 0
    
    if self.df.loc[idx, 'classID'] == 'adults':
        class_id = 1
    if self.df.loc[idx, 'classID'] == 'mixed':
        class_id = 2
        
    if self.df.loc[idx, 'classID'] == 'transitions':
        
        if self.args.overlap_class == False:
            class_id = 2
        else:
            class_id = 3

    aud = AudioUtil.open(audio_file)
    rechan = AudioUtil.rechannel(aud, self.channel)
    norm = AudioUtil.normalize(rechan)
    shift = AudioUtil.time_shift(norm, self.shift_pct)
    add_noise = AudioUtil.add_noise(shift,0.20)
    sgram = AudioUtil.spectro_gram(add_noise, n_mels=128, n_fft=2048, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.2, n_freq_masks=8, n_time_masks=8)

    return aug_sgram, class_id

  def spit_out(self, idx):
    audio_file = self.data_path + self.df.loc[idx, 'relative_path']
    Audio(filename=audio_file)
