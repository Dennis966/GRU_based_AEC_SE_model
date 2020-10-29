# PESQ
import subprocess
# STOI
import soundfile as sf
from pystoi.stoi import stoi
# SSNR
from SSNR_MODULE.Module import SSNR
# NCM
from NCM_MODULE.Module import NCM

import glob
from tqdm import tqdm
import csv
import numpy as np

from utils_SE_fixAEC import cal_time
import time

def scoring_PESQ(model_detail_time, voc_wav_path, NNvoc_wav_path, sr):
    # print('before replace voc_wav_path =', voc_wav_path)
    voc_wav_path = voc_wav_path.replace('(', '\(')
    voc_wav_path = voc_wav_path.replace(')', '\)')
    # print(' after replace voc_wav_path =', voc_wav_path)

    # print('before replace NNvoc_wav_path =', NNvoc_wav_path)
    NNvoc_wav_path = NNvoc_wav_path.replace('(', '\(')
    NNvoc_wav_path = NNvoc_wav_path.replace(')', '\)')
    # print(' after replace NNvoc_wav_path =', NNvoc_wav_path)

    # the PESQ execute file cannot handle files with long path strings
    # may output wrong and super low PESQ
    # so copy the files to the folder where the PESQ execute file is to reduce the path strings
    subprocess.check_output('cp %s voc_%s.wav' % (voc_wav_path, model_detail_time), shell=True)
    subprocess.check_output('cp %s NNvoc_%s.wav' % (NNvoc_wav_path, model_detail_time), shell=True)

    ci_pesq = subprocess.check_output('./PESQ +%d voc_%s.wav NNvoc_%s.wav' % (sr, model_detail_time, model_detail_time), shell=True)
    ci_pesq = ci_pesq.decode("utf-8")
    ci_pesq = ci_pesq.splitlines()[-1]
    ci_pesq = ci_pesq[-5:]
    ci_pesq = float(ci_pesq)

    subprocess.call('rm voc_%s.wav' % model_detail_time, shell=True)
    subprocess.call('rm NNvoc_%s.wav' % model_detail_time, shell=True)

    return ci_pesq

def scoring_STOI(voc, NNvoc, sr):
    ci_stoi = stoi(voc, NNvoc, sr, extended=False)

    return ci_stoi

def scoring_NCM(voc_wav_path, NNvoc_wav_path):
    ci_ncm = NCM().score(voc_wav_path, NNvoc_wav_path)

    return ci_ncm

def scoring_file(model_detail_time, voc_wav_path, NNvoc_wav_path):
    file_name = NNvoc_wav_path.rsplit('.', 1)[0]
    file_name = file_name.rsplit('/', 1)[-1]
    # print('file_name =', file_name)

    voc, sr = sf.read(voc_wav_path)
    NNvoc, sr = sf.read(NNvoc_wav_path)

    ci_pesq = scoring_PESQ(model_detail_time, voc_wav_path, NNvoc_wav_path, sr)
    ci_stoi = scoring_STOI(voc, NNvoc, sr)
    ci_ncm = scoring_NCM(voc_wav_path, NNvoc_wav_path)

    return file_name, ci_pesq, ci_stoi, ci_ncm

def scoring_dir(model_detail_time, scoring_path_list):
    scores = []

    for dir_path in scoring_path_list:
        voc_path, NNvoc_path = dir_path

        print('Scoring PESQ, STOI and NCM of wav in \'' + NNvoc_path + '\'...')
        for wav_path in tqdm(sorted(glob.glob(NNvoc_path + '*.wav'))):

            voc_wav_path   = wav_path.replace(NNvoc_path, voc_path)
            NNvoc_wav_path = wav_path

            scores.append(scoring_file(model_detail_time, voc_wav_path, NNvoc_wav_path))

    return scores # list of (file_name, ci_pesq, ci_stoi, ci_ncm)

def prepare_scoring_list(dataset_path, result_audio_path):
    from prepare_path_list_AEC import dataset, test_spk_list, noise_test_list, snr_test_list

    test_spk_list = [str(i).zfill(2) for i in test_spk_list]

    path_list = []

    if dataset == 'AV_enh':
        for spk_num in test_spk_list:
            # (voc_path, NNvoc_path)
            path_list.extend([
            (dataset_path + 'SP' + spk_num + '/audio_voc/clean/test/',
             result_audio_path + 'SP' + spk_num + '/clean/')
            ])

            ### NNvoc_noisy waveforms can not be the ref for scoring PESQ
            # for noise in noise_test_list:
            #     for snr in snr_test_list:
            #         noise_snrdb = noise + '_' + str(snr).replace('-', 'n') + 'db'
            #         # (voc_noisy_path, NNvoc_noisy_path)
            #         path_list.extend([
            #         (dataset_path + 'SP' + spk_num + '/audio_voc/noisy_enh/test/' + noise_snrdb + '/',
            #          result_audio_path + 'SP' + spk_num + '/' + noise_snrdb + '/')
            #         ])

    # print('path_list =', path_list)

    return path_list

def write_score(path_list, result_model_path):
    start_time = time.time()

    # print('result_model_path =', result_model_path)
    model_detail = result_model_path.rsplit('/', 2)[-2]
    # print('model_detail =', model_detail)

    model_detail_time = model_detail + str(start_time)
    model_detail_time = model_detail_time.replace('(', '\(')
    model_detail_time = model_detail_time.replace(')', '\)')

    scores = scoring_dir(model_detail_time, path_list)

    count = len(scores)
    sum_ci_pesq = 0.0
    sum_ci_stoi = 0.0
    sum_ci_ncm = 0.0

    # CSV Result Output
    f = open(result_model_path + 'Results_Report[%s].csv' % model_detail, 'w')
    w = csv.writer(f)
    w.writerow(('File_Name', 'CI_PESQ', 'CI_STOI', 'CI_NCM'))

    for score in scores:
        file_name, ci_pesq, ci_stoi, ci_ncm = score

        sum_ci_pesq += ci_pesq
        sum_ci_stoi += ci_stoi
        sum_ci_ncm += ci_ncm

        w.writerow((file_name, ci_pesq, ci_stoi, ci_ncm))

    print()
    w.writerow(())

    mean_ci_pesq = sum_ci_pesq / count
    mean_ci_stoi = sum_ci_stoi / count
    mean_ci_ncm = sum_ci_ncm / count

    print('mean_ci_pesq = %5.3f, mean_ci_stoi = %5.3f, mean_ci_ncm = %5.3f' % (mean_ci_pesq, mean_ci_stoi, mean_ci_ncm))
    print()

    w.writerow(())
    w.writerow(('total mean', mean_ci_pesq, mean_ci_stoi, mean_ci_ncm))
    f.close()

    # remove the files created during scoring PESQ
    # subprocess.call('rm voc_%s.wav' % model_detail_time, shell=True)
    # subprocess.call('rm NNvoc_%s.wav' % model_detail_time, shell=True)

    # remove the by-product created by the PESQ execute file
    subprocess.call(['rm', '_pesq_itu_results.txt'])
    subprocess.call(['rm', '_pesq_results.txt'])

    end_time = time.time()

    score_time = cal_time(start_time, end_time)
    print('Scoring complete.\n')

    return score_time
