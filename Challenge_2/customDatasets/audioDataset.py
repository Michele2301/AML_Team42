from torch.utils.data import DataLoader, Dataset, random_split
from utils.audioUtils import AudioUtil
from torchaudio import transforms
import torch

class AudioDataset(Dataset):
    def __init__(self, df, data_path,transforms=None, in_memory=False, sgram_type="mel", augment=False, split_sgram=False, test_mode=False):
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
        self.in_memory = in_memory
        self.sgram_type = sgram_type
        self.augment = augment
        self.split_sgram = split_sgram
        self.test_mode = test_mode
        self.mean=None
        self.std=None

        # Save all audio already in memory, so that getitem does not have to read every time from disk
        if in_memory:
            self.audios = {}
            for idx in range(len(df)):
                audio_file = self.data_path + self.df.loc[idx, 'filename']
                aud = AudioUtil.open(audio_file)
                self.audios[idx] = aud


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.data_path + self.df.loc[idx, 'filename']
        label = self.df.loc[idx, 'is_normal']
        id = self.df.loc[idx, 'machine_id']

        if self.in_memory:
            aud = self.audios[idx]
        else:
            aud = AudioUtil.open(audio_file)

        dur_aud = AudioUtil.pad_trunc(aud, self.duration)

        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=128, n_fft=1000, hop_len=501, sgram_type=self.sgram_type)
        aug_sgram = sgram
        
        if self.augment:
            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        aug_sgram=aug_sgram.mT

        #To use in test mode when using different frames for the same sample
        if self.test_mode:
            shape = aug_sgram.shape[1]
            aug_sgrams = []
            for idx in range(10):
                len = int(shape / 10)
                new_aug_sgram = aug_sgram[:, len*idx:len*idx + len, :]
                # normalize mean 0 var 1
                if self.mean is not None and self.std is not None:
                    new_aug_sgram = (new_aug_sgram - self.mean) / self.std
                if self.min is not None and self.max is not None:
                    new_aug_sgram = (new_aug_sgram-self.min)/(self.max-self.min)
                aug_sgrams.append(new_aug_sgram)
            return torch.cat(aug_sgrams), label

        #To use in training when using different frames for the same sample
        if self.split_sgram:
            idx = torch.randint(10, (1,)).item()
            len = int(aug_sgram.shape[1] / 10)
            aug_sgram = aug_sgram[:, len*idx:len*idx + len, :]

        if self.mean is not None and self.std is not None:
            aug_sgram = (aug_sgram-self.mean)/self.std

        if self.min is not None and self.max is not None:
            aug_sgram = (aug_sgram-self.min)/(self.max-self.min)

        if self.with_id:
            return aug_sgram, label, id
        
        if self.with_filename:
            return aug_sgram, label, self.df.loc[idx, 'filename']
        

        return aug_sgram, label
