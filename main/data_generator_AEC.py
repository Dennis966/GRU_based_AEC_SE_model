import torch
import torchaudio
from torch.utils.data import Dataset
import glob
import random

class wav_Dataset(Dataset):
    def __init__(self, data_path_list, mode):
        self.mode = mode # mode: 'training', 'validation', 'testing'
        self.samples = []

        for dir_path in data_path_list:
            path, ci_path = dir_path

            wav_paths = sorted(glob.glob(path + '*.wav'))

            for wav_path in wav_paths:
                file_name = wav_path.rsplit('.', 1)[0]
                file_name = file_name.rsplit('/', 2)[-1]

                ci_wav_path = wav_path.replace(path, ci_path)

                self.samples.append((wav_path, '', ci_wav_path))

        if self.mode == 'training':
            random.shuffle(self.samples)
            self.samples = self.samples[:12000]
        elif self.mode == 'validation':
            random.shuffle(self.samples)
            self.samples = self.samples[:800]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, _, ci_wav_path = self.samples[idx]
        # print('wav_path =', wav_path)

        file_name = wav_path.rsplit('.', 1)[0]
        file_name = file_name.rsplit('/', 2)
        noise_type = file_name[-2]

        if noise_type in ['train', 'val', 'test']:
            noise_type = 'clean'

        file_name = file_name[-1]
        # print('file_name =', file_name)
        # print('noise_type =', noise_type)
        file_name = file_name + '__' + noise_type
        # print('file_name =', file_name)

        wav, _ = torchaudio.load(wav_path)
        wav_l = wav.shape[-1]

        if self.mode == 'training' or self.mode == 'validation':
            wav_ci, _ = torchaudio.load(ci_wav_path)
            wav_ci_l = wav_ci.shape[-1]

            if wav_l != wav_ci_l:
                raise ValueError('wav and wav_ci should have the same length.')

        elif self.mode == 'testing':
            wav_ci = None

        # (file_name, wav_length, wav, wav_noisy, wav_ci)
        data = file_name, wav_l, wav, wav_ci

        return data
