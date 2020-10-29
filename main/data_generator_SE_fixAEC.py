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
            _, noisy_path, clean_ci_path = dir_path

            wav_paths = sorted(glob.glob(noisy_path + '*.wav'))

            for wav_path in wav_paths:
                file_name = wav_path.rsplit('.', 1)[0]
                file_name = file_name.rsplit('/', 2)[-1]

                noisy_wav_path = wav_path
                clean_ci_wav_path = wav_path.replace(noisy_path, clean_ci_path)

                self.samples.append(('', noisy_wav_path, clean_ci_wav_path))

        if self.mode == 'training':
            random.shuffle(self.samples)
            self.samples = self.samples[:12000]
        elif self.mode == 'validation':
            random.shuffle(self.samples)
            self.samples = self.samples[:800]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, noisy_wav_path, clean_ci_wav_path = self.samples[idx]
        # print('noisy_wav_path =', noisy_wav_path)

        file_name = noisy_wav_path.rsplit('.', 1)[0]
        file_name = file_name.rsplit('/', 2)
        noisy_type = file_name[-2]
        file_name = file_name[-1]
        # print('file_name =', file_name)
        # print('noisy_type =', noisy_type)
        file_name = file_name + '__' + noisy_type
        # print('file_name =', file_name)

        wav_noisy, _ = torchaudio.load(noisy_wav_path)
        # print('wav_noisy.shape =', wav_noisy.shape)
        wav_noisy_l = wav_noisy.shape[-1]
        wav_noisy /= torch.max(torch.abs(wav_noisy))
        # print('wav_noisy.shape =', wav_noisy.shape)

        if self.mode == 'training' or self.mode == 'validation':
            wav_clean_ci, _ = torchaudio.load(clean_ci_wav_path)
            wav_clean_ci_l = wav_clean_ci.shape[-1]

            if wav_noisy_l != wav_clean_ci_l:
                raise ValueError('noisy and clean_ci wav should have the same length.')

        elif self.mode == 'testing':
            wav_clean_ci = None

        # (file_name, wav_length, wav_clean, wav_noisy, wav_clean_ci)
        data = file_name, wav_noisy_l, None, wav_noisy, wav_clean_ci

        return data
