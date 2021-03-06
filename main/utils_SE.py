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
from oct2py import octave

num_workers = 8
pin_memory = True
# print()
# print('num_workers =', num_workers)

n_fft = 512
hop_length = 256
win_length = 512
window = torch.hann_window(win_length)

octave.eval('pkg load signal')
octave.addpath(os.getcwd()+"/CI_vocoder")

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
    file_name, wav_length, clean, noisy, _ = list(zip(*batch))

    max_wl = max(wav_length)
    max_wl = win_length * (max_wl // win_length + 1)

    wav_length = torch.IntTensor(wav_length)

    trim_clean = True if isinstance(clean[0], torch.Tensor) else False

    noisy = list(noisy)
    clean = list(clean)

    # print()
    # print('before trim.')
    # print('noisy[0].shape =', noisy[0].shape)

    # trimming each data in a batch to the length of max_wl
    for i, _ in enumerate(noisy):
        wav_length_i = noisy[i].shape[-1]
        pad_length = max_wl - wav_length_i

        noisy[i] = F.pad(noisy[i], (0, pad_length), mode='constant', value=0)

        if trim_clean:
            clean[i] = F.pad(clean[i], (0, pad_length), mode='constant', value=0)

    # print()
    # print(' after trim.')
    # print('noisy[0].shape =', noisy[0].shape)

    clean = torch.stack(clean, 0) if trim_clean else None
    noisy = torch.stack(noisy, 0)

    return file_name, wav_length, clean, noisy, None

# Section of training
def train(device, model, train_dataset, val_dataset, epochs, train_batch_size, criterion, optimizer, scheduler, result_model_path):
    model.train()
    model_name = model.__class__.__name__
    lr = optimizer.defaults['lr']

    model_detail = model_detail_string(model_name, epochs, lr, train_batch_size)

    loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
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
                #                 [2] wav_clean or None
                #                 [3] wav_noisy
                #                 [4] wav_clean_ci
                file_name, _, wav_clean, wav_noisy, _ = batch
                # print('file_name =', file_name)

                wav_clean = wav_clean.to(device, non_blocking=True)
                wav_noisy = wav_noisy.to(device, non_blocking=True)

                spec_clean, _ = wav2spec(wav_clean)
                spec_noisy, _ = wav2spec(wav_noisy)

                spec_enhan = model(spec_noisy)
                loss = criterion(spec_enhan, spec_clean)
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

    loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    total_step = len(loader)

    val_loss = 0.0

    with torch.no_grad():
        with tqdm(total=total_step, desc='                  val_loss: %7.4f' % val_loss,
                  dynamic_ncols=True, bar_format='{l_bar} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}]') as t:
            for step, batch in enumerate(loader):
                # data structure: [0] file_name
                #                 [1] wav_length
                #                 [2] wav_clean or None
                #                 [3] wav_noisy
                #                 [4] wav_clean_ci
                file_name, _, wav_clean, wav_noisy, _ = batch
                # print('file_name =', file_name)

                wav_clean = wav_clean.to(device, non_blocking=True)
                wav_noisy = wav_noisy.to(device, non_blocking=True)

                spec_clean, _ = wav2spec(wav_clean)
                spec_noisy, _ = wav2spec(wav_noisy)

                spec_enhan = model(spec_noisy)
                loss = criterion(spec_enhan, spec_clean)
                val_loss += loss.item() / total_step

                t.set_description('                  val_loss: %7.4f' % val_loss) # space for alignment of tqdm
                t.update()

    model.train()

    return val_loss

# Section of testing
def test(device, model, dataset, test_batch_size, result_audio_path, result_audio_voc_path):
    model.eval()
    model_name = model.__class__.__name__

    mode = dataset.mode

    loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    total_step = len(loader)

    start_time = time.time()
    print(time.asctime(time.localtime(time.time())) + ' Start testing ' + model_name + '...')
    print()

    print('Saving result wav to \'' + result_audio_path +     '\'\n' + \
          '                 and \'' + result_audio_voc_path + '\'...')

    with torch.no_grad():
        with tqdm(total=total_step, dynamic_ncols=True, bar_format='{l_bar} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}]') as t:
            for step, batch in enumerate(loader):
                file_names, wav_lengths, _, wav_noisys, _ = batch
                # print('file_names =', file_names)

                wav_noisys = wav_noisys.to(device, non_blocking=True)

                spec_noisys, phases = wav2spec(wav_noisys)

                spec_enhans = model(spec_noisys)

                wav_enhans = spec2wav(spec_enhans, phases)
                wav_enhans = wav_enhans.cpu()

                outputs = zip(file_names, wav_lengths, wav_enhans)

                for output in outputs:
                    file_name, wav_length, wav_enhan = output
                    # print('file_name =', file_name)

                    file_name = file_name.split('__')
                    noise_type = file_name[1]
                    # print('noise_type =', noise_type)
                    file_name = file_name[0]
                    # print('file_name =', file_name)
                    spk_name = file_name.split('_')[0]

                    wav_enhan = wav_enhan[..., :wav_length]
                    wav_enhan /= torch.max(torch.abs(wav_enhan))
                    wav_enhan /= 8

                    result_enhan_dir_path = result_audio_path + spk_name + '/' + noise_type + '/'
                    result_enhan_path = result_enhan_dir_path + file_name + '.wav'
                    if not os.path.exists(result_enhan_dir_path):
                        os.makedirs(result_enhan_dir_path)

                    torchaudio.save(result_enhan_path, wav_enhan, 16000, precision=16)

                    result_ci_dir_path = result_audio_voc_path + spk_name + '/' + noise_type + '/'
                    result_ci_path = result_ci_dir_path + file_name + '.wav'
                    if not os.path.exists(result_ci_dir_path):
                        os.makedirs(result_ci_dir_path)

                    octave.wav2vocwav(result_enhan_path, result_ci_path)

                t.set_description('output wavs: %3d/%3d' % ((step + 1), total_step))
                t.update()

    print()
    print(time.asctime(time.localtime(time.time())) + ' Testing complete.')
    end_time = time.time()

    test_time = cal_time(start_time, end_time)
    print ('Model tested for %2d day %2d hr %2d min.\n' % test_time)

    return test_time
