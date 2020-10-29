import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from visdom import Visdom
import time
import numpy as np
from visdom_save import vis

import pdb

num_workers = 8
pin_memory = True
# print()
# print('num_workers =', num_workers)

n_fft = 512
hop_length = 256
win_length = 512
window = torch.hann_window(win_length)

def model_detail_string(model_name, epochs, lr, train_batch_size):
    model_detail = '%s_e(%d)lr(%.0e)bs(%d)' % (model_name, epochs, lr, train_batch_size)

    return model_detail

def cal_time(start_time, end_time):
    s = end_time - start_time
    m = s // 60
    h = m // 60
    m = m % 60
    d = h // 24
    h = h % 24

    if s >= 30:
        m += 1
        if m == 60:
            h += 1
            if h == 24:
                d += 1

    return d, h, m

def wav2spec(wav):
    # print('wav.shape =', wav.shape) # (bs, 1, len)
    wav = wav.squeeze(1)
    # print('wav.shape =', wav.shape) # (bs, len)

    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window.to(wav.device),
                      pad_mode='reflect', normalized=False, onesided=True)
    # print('stft.shape =', stft.shape) # (bs, 257, frame_len, 2)

    spec = torch.norm(stft, 2, -1)

    phase = stft / (spec.unsqueeze(-1) + 1e-12)
    spec = torch.log10(spec + 1) # log1p
    # print('spec.shape =', spec.shape) # (bs, 257, frame_len)

    return spec, phase

def spec2wav(spec, phase):
    # print('spec.shape =', spec.shape) # (bs, 257, frame_len)
    # print('phase.shape =', phase.shape) # (bs, 257, frame_len, 2)

    spec = 10 ** spec - 1 # inverse log1p
    stft = (spec.unsqueeze(-1) - 1e-12) * phase
    # print('stft.shape =', stft.shape) # (bs, 257, frame_len, 2)

    wav = torchaudio.functional.istft(stft, n_fft=n_fft, hop_length=hop_length,
                                      win_length=win_length, window=window.to(stft.device),
                                      center=True, pad_mode='reflect', normalized=False,
                                      onesided=True, length=None)
    # print('wav.shape =', wav.shape) # (bs, len)
    wav = wav.unsqueeze(1)
    # print('wav.shape =', wav.shape) # (bs, 1, len)

    return wav

# Section of functions for DataLoader
def collate_fn(batch):
    file_name, wav_length, wav, wav_ci = list(zip(*batch))

    max_wl = max(wav_length)
    max_wl = win_length * (max_wl // win_length + 1)
    # print('max_wl =', max_wl)
    wav_length = torch.IntTensor(wav_length)
    # print('wav_length =', wav_length)

    trim_wav_ci = True if isinstance(wav_ci[0], torch.Tensor) else False

    wav = list(wav)
    wav_ci = list(wav_ci)

    # print()
    # print('before pad.')
    # print('before wav[0].shape =', wav[0].shape)
    # print('before wav[1].shape =', wav[1].shape)
    # print('before wav[0][0] = ', wav[0][0])
    # print('before wav[1][0] = ', wav[1][0])

    # padding each data in a batch to the length of max_wl
    for i, _ in enumerate(wav):
        wav_length_i = wav[i].shape[-1]
        pad_length = max_wl - wav_length_i

        wav[i] = F.pad(wav[i], (0, pad_length), mode='constant', value=0)

        if trim_wav_ci:
            wav_ci[i] = F.pad(wav_ci[i], (0, pad_length), mode='constant', value=0)

    # print()
    # print(' after pad.')
    # print(' after wav[0].shape =', wav[0].shape)
    # print(' after wav[1].shape =', wav[1].shape)
    # print(' after wav[0][0] = ', wav[0][0])
    # print(' after wav[1][0] = ', wav[1][0])

    wav = torch.stack(wav, 0)
    wav_ci = torch.stack(wav_ci, 0) if trim_wav_ci else None

    return file_name, wav_length, wav, wav_ci

# Section of training
def train(device, model, train_dataset, val_dataset, epochs, train_batch_size, criterion, optimizer, scheduler, result_model_path):
    model.train()
    model_name = model.__class__.__name__
    lr = optimizer.defaults['lr']

    model_detail = model_detail_string(model_name, epochs, lr, train_batch_size)

    loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=True)
    total_step = len(loader)
    # print('total_step =', total_step)

    patience = 10 # how many epochs training will automatically stop if val_loss not improves

    print('Visdom activating.')
    viz = Visdom(env='%s' % model_detail)
    print()

    start_time = time.time()
    print(time.asctime(time.localtime(time.time())) + ' Start training ' + model_detail + '...')
    print()

    for epoch in range(epochs):
        running_loss = 0.0

        optimizer.zero_grad()

        with tqdm(total=total_step, desc='epoch: %3d/%3d, train_loss: %7.4f' % (epoch + 1, epochs, running_loss),
                  dynamic_ncols=True, bar_format='{l_bar} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}]') as t:
            for step, batch in enumerate(loader):
                # data structure: [0] file_name
                #                 [1] wav_length
                #                 [2] wav_clean or wav_noisy or None
                #                 [3] _
                #                 [4] wav_clean_ci or wav_noisy_ci
                file_name, wav_length, wav, wav_ci = batch
                # print('file_name =', file_name)

                wav = wav.to(device, non_blocking=True)
                wav_ci = wav_ci.to(device, non_blocking=True)

                spec, _ = wav2spec(wav)
                spec_ci, _ = wav2spec(wav_ci)
                # clean_ci, _ = wav2spec(clean_ci)

                spec_NNci = model(spec)
                loss = criterion(spec_NNci, spec_ci)
                loss.backward()
                running_loss += loss.item() / total_step

                optimizer.step()
                optimizer.zero_grad()

                t.set_description('epoch: %3d/%3d, train_loss: %7.4f' % (epoch + 1, epochs, running_loss))
                t.update()

        # print('old lr = %.0e' % optimizer.param_groups[0]['lr'])
        scheduler.step()
        # print('new lr = %.0e' % optimizer.param_groups[0]['lr'])

        val_loss = val(device, model, val_dataset, train_batch_size, criterion)

        viz.line(
            np.column_stack((running_loss, val_loss)), np.column_stack((epoch + 1, epoch + 1)), win='loss', update='append',
            opts=dict(title='loss', xlabel='epoch', ylabel='loss', showlegend=True, legend=['training', 'val'])
        )

        ### save best model ###
        # init
        if epoch == 0:
            running_patience = patience # running_patience init
            best_loss = val_loss
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': running_loss,
                        'val_loss': val_loss}, result_model_path + 'best_model[%s].tar' % model_detail)
            # print('init model saved.\n')

        # update
        if val_loss < best_loss:
            running_patience = patience # running_patience reset
            best_loss = val_loss
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': running_loss,
                        'val_loss': val_loss}, result_model_path + 'best_model[%s].tar' % model_detail)
            # print('best model saved.\n')
        else:
            running_patience -= 1
            if running_patience == 0:
                print('training stopped since val_loss did not improve in the last %d epochs.' % patience)
                break

    torch.save({'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': running_loss,
                'val_loss': val_loss}, result_model_path + 'model[%s].tar' % model_detail)

    # save visdom loss curves
    print()
    print('Visdom saving.')
    result_visdom_path = result_model_path + 'visdom_loss[%s].log' % model_detail
    vis.create_log_at(result_visdom_path, model_detail)
    print()

    print(time.asctime(time.localtime(time.time())) + ' Training complete.')
    end_time = time.time()

    train_time = cal_time(start_time, end_time)
    print ('Model trained for %2d day %2d hr %2d min.\n' % train_time)

    return train_time

def val(device, model, val_dataset, train_batch_size, criterion):
    model.eval()

    loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=True)
    total_step = len(loader)

    val_loss = 0.0

    with torch.no_grad():
        with tqdm(total=total_step, desc='                  val_loss: %7.4f' % val_loss,
                  dynamic_ncols=True, bar_format='{l_bar} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}]') as t:
            for step, batch in enumerate(loader):
                # data structure: [0] file_name
                #                 [1] wav_length
                #                 [2] wav_clean or wav_noisy or None
                #                 [3] _
                #                 [4] wav_clean_ci or wav_noisy_ci
                file_name, wav_length, wav, wav_ci = batch
                # print('file_name =', file_name)

                wav = wav.to(device, non_blocking=True)
                wav_ci = wav_ci.to(device, non_blocking=True)

                spec, _ = wav2spec(wav)
                spec_ci, _ = wav2spec(wav_ci)

                spec_NNci = model(spec)
                loss = criterion(spec_NNci, spec_ci)
                val_loss += loss.item() / total_step

                t.set_description('                  val_loss: %7.4f' % val_loss) # space for alignment of tqdm
                t.update()

    model.train()

    return val_loss

# Section of testing
def test(device, model, dataset, test_batch_size, result_audio_path):
    model.eval()
    model_name = model.__class__.__name__

    mode = dataset.mode

    loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    total_step = len(loader)

    start_time = time.time()
    print(time.asctime(time.localtime(time.time())) + ' Start testing ' + model_name + '...')
    print()

    print('Saving result wav to \'' + result_audio_path +     '\'...')

    with torch.no_grad():
        with tqdm(total=total_step, dynamic_ncols=True, bar_format='{l_bar} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}]') as t:
            for step, batch in enumerate(loader):
                file_names, wav_lengths, wavs, _ = batch
                # print('file_names =', file_names)

                wavs = wavs.to(device, non_blocking=True)

                specs, phases = wav2spec(wavs)

                spec_NNcis = model(specs)

                wav_NNcis = spec2wav(spec_NNcis, phases)
                wav_NNcis = wav_NNcis.cpu()

                outputs = zip(file_names, wav_lengths, wav_NNcis)

                for output in outputs:
                    file_name, wav_length, wav_NNci = output
                    # print('file_name =', file_name)

                    file_name = file_name.split('__')
                    noise_type = file_name[1]
                    # print('noise_type =', noise_type)
                    file_name = file_name[0]
                    # print('file_name =', file_name)
                    spk_name = file_name.split('_')[0]

                    wav_NNci = wav_NNci[..., :wav_length]
                    wav_NNci /= torch.max(torch.abs(wav_NNci))
                    wav_NNci /= 8

                    result_ci_dir_path = result_audio_path + spk_name + '/' + noise_type + '/'
                    result_ci_path = result_ci_dir_path + file_name + '.wav'
                    if not os.path.exists(result_ci_dir_path):
                        os.makedirs(result_ci_dir_path)

                    torchaudio.save(result_ci_path, wav_NNci, 16000, precision=16)

                t.set_description('output wavs: %3d/%3d' % ((step + 1), total_step))
                t.update()

    print()
    print(time.asctime(time.localtime(time.time())) + ' Testing complete.')
    end_time = time.time()

    test_time = cal_time(start_time, end_time)
    print ('Model tested for %2d day %2d hr %2d min.\n' % test_time)

    return test_time
