from torch.utils.data import DataLoader, Dataset, random_split
from utils.audioUtils import AudioUtil
from torchvision import transforms
import torch

class AudioDataset(Dataset):
    def __init__(self, df, data_path,transforms=None):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 10_000
        self.sr = 16_000
        self.channel = 2
        self.shift_pct = 0.4
        self.min=None
        self.max=None
        self.with_id=False
        self.with_filename=False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.data_path + self.df.loc[idx, 'filename']
        label = self.df.loc[idx, 'is_normal']
        id = self.df.loc[idx, 'machine_id']

        aud = AudioUtil.open(audio_file)
        dur_aud = AudioUtil.pad_trunc(aud, self.duration)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=128, n_fft=1000, hop_len=501)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        aug_sgram=aug_sgram.mT
        if self.min is not None and self.max is not None:
            aug_sgram = (aug_sgram-self.min)/(self.max-self.min)
        if self.with_id:
            return aug_sgram, label, id
        if self.with_filename:
            return aug_sgram, label, self.df.loc[idx, 'filename']
        return aug_sgram, label