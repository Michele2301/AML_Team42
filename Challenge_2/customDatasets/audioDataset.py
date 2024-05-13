from torch.utils.data import DataLoader, Dataset, random_split
from utils.audioUtils import AudioUtil

class AudioDataset(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 10_000
        self.sr = 16_000
        self.channel = 2
        self.shift_pct = 0.4
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        audio_file = self.data_path + self.df.loc[idx, 'filename']
        label = self.df.loc[idx, 'normal']
        
        aud = AudioUtil.open(audio_file)
        dur_aud = AudioUtil.pad_trunc(aud, self.duration)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_mask=2, n_time_masks=2)

        return aug_sgram, label