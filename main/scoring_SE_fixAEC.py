# PESQ
import subprocess
# STOI
import soundfile as sf
from pystoi.stoi import stoi
# NCM
from NCM_MODULE.Module import NCM

import glob
from tqdm import tqdm
import csv
from visdom import Visdom
from visdom_save import vis
import numpy as np

from utils_SE_fixAEC import cal_time
import time

def scoring_PESQ(model_detail_time, clean_wav_path, noisy_wav_path, enhan_wav_path, sr):
    # print('before replace clean_wav_path =', clean_wav_path)
    clean_wav_path = clean_wav_path.replace('(', '\(')
    clean_wav_path = clean_wav_path.replace(')', '\)')
    # print(' after replace clean_wav_path =', clean_wav_path)

    # print('before replace noisy_wav_path =', noisy_wav_path)
    noisy_wav_path = noisy_wav_path.replace('(', '\(')
    noisy_wav_path = noisy_wav_path.replace(')', '\)')
    # print(' after replace noisy_wav_path =', noisy_wav_path)

    # print('before replace enhan_wav_path =', enhan_wav_path)
    enhan_wav_path = enhan_wav_path.replace('(', '\(')
    enhan_wav_path = enhan_wav_path.replace(')', '\)')
    # print(' after replace enhan_wav_path =', enhan_wav_path)

    # the PESQ execute file cannot handle files with long path strings
    # may output wrong and super low PESQ
    # so copy the files to the folder where the PESQ execute file is to reduce the path strings
    subprocess.check_output('cp %s cln_%s.wav' % (clean_wav_path, model_detail_time), shell=True)
    subprocess.check_output('cp %s nsy_%s.wav' % (noisy_wav_path, model_detail_time), shell=True)
    subprocess.check_output('cp %s enh_%s.wav' % (enhan_wav_path, model_detail_time), shell=True)

    noisy_pesq = subprocess.check_output('./PESQ +%d cln_%s.wav nsy_%s.wav' % (sr, model_detail_time, model_detail_time), shell=True)
    noisy_pesq = noisy_pesq.decode("utf-8")
    noisy_pesq = noisy_pesq.splitlines()[-1]
    noisy_pesq = noisy_pesq[-5:]
    noisy_pesq = float(noisy_pesq)

    enhan_pesq = subprocess.check_output('./PESQ +%d cln_%s.wav enh_%s.wav' % (sr, model_detail_time, model_detail_time), shell=True)
    enhan_pesq = enhan_pesq.decode("utf-8")
    enhan_pesq = enhan_pesq.splitlines()[-1]
    enhan_pesq = enhan_pesq[-5:]
    enhan_pesq = float(enhan_pesq)

    return noisy_pesq, enhan_pesq

def scoring_STOI(clean, noisy, enhan, sr):
    noisy_stoi = stoi(clean, noisy, sr, extended=False)
    enhan_stoi = stoi(clean, enhan, sr, extended=False)

    return noisy_stoi, enhan_stoi

def scoring_NCM(clean_wav_path, noisy_wav_path, enhan_wav_path):
    noisy_ncm = NCM().score(clean_wav_path, noisy_wav_path)
    enhan_ncm = NCM().score(clean_wav_path, enhan_wav_path)

    return noisy_ncm, enhan_ncm

def scoring_file(model_detail_time, clean_wav_path, noisy_wav_path, enhan_wav_path, vocoded_clean_wav_path, vocoded_noisy_wav_path, vocoded_enhan_wav_path):

    file_name = noisy_wav_path.rsplit('.', 1)[0]
    file_name = file_name.rsplit('/', 2)
    noise_type = file_name[-2]
    file_name = file_name[-1]

    # print('noise_type =', noise_type)
    # print('file_name =', file_name)

    file_name = file_name + '__' + noise_type
    # print('file_name =', file_name)

    clean, sr = sf.read(clean_wav_path)
    noisy, sr = sf.read(noisy_wav_path)
    enhan, sr = sf.read(enhan_wav_path)

    vocoded_clean, sr = sf.read(vocoded_clean_wav_path)
    vocoded_noisy, sr = sf.read(vocoded_noisy_wav_path)
    vocoded_enhan, sr = sf.read(vocoded_enhan_wav_path)

    noisy_pesq, enhan_pesq = scoring_PESQ(model_detail_time, clean_wav_path, noisy_wav_path, enhan_wav_path, sr)
    noisy_stoi, enhan_stoi = scoring_STOI(clean, noisy, enhan, sr)
    noisy_ncm, enhan_ncm = scoring_NCM(clean_wav_path, noisy_wav_path, enhan_wav_path)

    vocoded_noisy_pesq, vocoded_enhan_pesq = scoring_PESQ(model_detail_time, vocoded_clean_wav_path, vocoded_noisy_wav_path, vocoded_enhan_wav_path, sr)
    vocoded_noisy_stoi, vocoded_enhan_stoi = scoring_STOI(vocoded_clean, vocoded_noisy, vocoded_enhan, sr)
    vocoded_noisy_ncm, vocoded_enhan_ncm = scoring_NCM(vocoded_clean_wav_path, vocoded_noisy_wav_path, vocoded_enhan_wav_path)

    return file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi, noisy_ncm, enhan_ncm, vocoded_noisy_pesq, vocoded_enhan_pesq, vocoded_noisy_stoi, vocoded_enhan_stoi, vocoded_noisy_ncm, vocoded_enhan_ncm

def scoring_dir(model_detail_time, scoring_path_list):
    scores = []

    for dir_path in scoring_path_list:
        clean_path, noisy_path, enhan_path, vocoded_clean_path, vocoded_noisy_path, vocoded_enhan_path = dir_path

        print('Scoring PESQ, STOI and NCM of wav in \'' + noisy_path         + '\'\n' + \
              '                                 and \'' + enhan_path         + '\',\n' + \
              '        PESQ, STOI and NCM of wav in \'' + vocoded_noisy_path + '\'\n' + \
              '                                 and \'' + vocoded_enhan_path + '\'...')
        for wav_path in tqdm(sorted(glob.glob(enhan_path + '*.wav'))):

            clean_wav_path         = wav_path.replace(enhan_path, clean_path)
            noisy_wav_path         = wav_path.replace(enhan_path, noisy_path)
            enhan_wav_path         = wav_path
            vocoded_clean_wav_path = wav_path.replace(enhan_path, vocoded_clean_path)
            vocoded_noisy_wav_path = wav_path.replace(enhan_path, vocoded_noisy_path)
            vocoded_enhan_wav_path = wav_path.replace(enhan_path, vocoded_enhan_path)

            # print('clean_wav_path =', clean_wav_path)
            # print('noisy_wav_path =', noisy_wav_path)
            # print('enhan_wav_path =', enhan_wav_path)
            scores.append(scoring_file(model_detail_time, clean_wav_path, noisy_wav_path, enhan_wav_path, vocoded_clean_wav_path, vocoded_noisy_wav_path, vocoded_enhan_wav_path))

    return scores # list of (file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi, noisy_ncm, enhan_ncm, vocoded_noisy_pesq, vocoded_enhan_pesq, vocoded_noisy_stoi, vocoded_enhan_stoi, vocoded_noisy_ncm, vocoded_enhan_ncm)

def prepare_scoring_list(dataset_path, result_audio_path, result_audio_voc_path):
    from prepare_path_list_SE_fixAEC import dataset, test_spk_list, noise_test_list, snr_test_list

    test_spk_list = [str(i).zfill(2) for i in test_spk_list]

    path_list = []

    if dataset == 'AV_enh':
        for spk_num in test_spk_list:
            for noise in noise_test_list:
                for snr in snr_test_list:
                    noise_snrdb = noise + '_' + str(snr).replace('-', 'n') + 'db'
                    # (clean_path, noisy_path, enhan_path, voc_clean_path, voc_noisy_path, voc_enhan_path)
                    path_list.extend([
                    (dataset_path + 'SP' + spk_num + '/audio/clean/test/',
                     dataset_path + 'SP' + spk_num + '/audio/noisy_enh/test/' + noise_snrdb + '/',
                     result_audio_path + 'SP' + spk_num + '/' + noise_snrdb + '/',
                     dataset_path + 'SP' + spk_num + '/audio_voc/clean/test/',
                     dataset_path + 'SP' + spk_num + '/audio_voc/noisy_enh/test/' + noise_snrdb + '/',
                     result_audio_voc_path + 'SP' + spk_num + '/' + noise_snrdb + '/')
                    ])

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
    sum_noisy_pesq = 0.0
    sum_enhan_pesq = 0.0
    sum_noisy_stoi = 0.0
    sum_enhan_stoi = 0.0
    sum_noisy_ncm = 0.0
    sum_enhan_ncm = 0.0
    sum_voc_noisy_pesq = 0.0
    sum_voc_enhan_pesq = 0.0
    sum_voc_noisy_stoi = 0.0
    sum_voc_enhan_stoi = 0.0
    sum_voc_noisy_ncm = 0.0
    sum_voc_enhan_ncm = 0.0

    # CSV Result Output
    f = open(result_model_path + 'Results_Report[%s].csv' % model_detail, 'w')
    w = csv.writer(f)
    w.writerow(('File_Name', 'Noisy_PESQ', 'Enhan_PESQ', 'Noisy_STOI', 'Enhan_STOI', 'Noisy_NCM', 'Enhan_NCM', 'Voc_Noisy_PESQ', 'Voc_Enhan_PESQ', 'Voc_Noisy_STOI', 'Voc_Enhan_STOI', 'Voc_Noisy_NCM', 'Voc_Enhan_NCM'))

    db_score = {}
    db_order = []
    db_noisetype_score = {}
    db_noisetype_order = []
    noisetype_order = []

    for score in scores:
        file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi, noisy_ncm, enhan_ncm, vocoded_noisy_pesq, vocoded_enhan_pesq, vocoded_noisy_stoi, vocoded_enhan_stoi, vocoded_noisy_ncm, vocoded_enhan_ncm = score

        sum_noisy_pesq += noisy_pesq
        sum_enhan_pesq += enhan_pesq
        sum_noisy_stoi += noisy_stoi
        sum_enhan_stoi += enhan_stoi
        sum_noisy_ncm += noisy_ncm
        sum_enhan_ncm += enhan_ncm
        sum_voc_noisy_pesq += vocoded_noisy_pesq
        sum_voc_enhan_pesq += vocoded_enhan_pesq
        sum_voc_noisy_stoi += vocoded_noisy_stoi
        sum_voc_enhan_stoi += vocoded_enhan_stoi
        sum_voc_noisy_ncm += vocoded_noisy_ncm
        sum_voc_enhan_ncm += vocoded_enhan_ncm

        w.writerow((file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi, noisy_ncm, enhan_ncm, vocoded_noisy_pesq, vocoded_enhan_pesq, vocoded_noisy_stoi, vocoded_enhan_stoi, vocoded_noisy_ncm, vocoded_enhan_ncm))

        db = file_name.rsplit('_', 1)[-1]
        # print('db =', db)
        db_noisetype = file_name.rsplit('__', 1)[-1]
        # print('db_noisetype =', db_noisetype)
        noisetype = db_noisetype.rsplit('_', 1)[-2]
        # print('noisetype =', noisetype)

        if db in db_score:
            # print('update ' + db)
            db_score[db]['db_count'] += 1
            db_score[db]['db_sum_noisy_pesq'] += noisy_pesq
            db_score[db]['db_sum_enhan_pesq'] += enhan_pesq
            db_score[db]['db_sum_noisy_stoi'] += noisy_stoi
            db_score[db]['db_sum_enhan_stoi'] += enhan_stoi
            db_score[db]['db_sum_noisy_ncm'] += noisy_ncm
            db_score[db]['db_sum_enhan_ncm'] += enhan_ncm
            db_score[db]['db_sum_voc_noisy_pesq'] += vocoded_noisy_pesq
            db_score[db]['db_sum_voc_enhan_pesq'] += vocoded_enhan_pesq
            db_score[db]['db_sum_voc_noisy_stoi'] += vocoded_noisy_stoi
            db_score[db]['db_sum_voc_enhan_stoi'] += vocoded_enhan_stoi
            db_score[db]['db_sum_voc_noisy_ncm'] += vocoded_noisy_ncm
            db_score[db]['db_sum_voc_enhan_ncm'] += vocoded_enhan_ncm
        else:
            # print('add ' + db)
            db_order.append(db)
            db_score[db] = {'db_count': 1,
                            'db_sum_noisy_pesq': noisy_pesq, 'db_sum_enhan_pesq': enhan_pesq,
                            'db_sum_noisy_stoi': noisy_stoi, 'db_sum_enhan_stoi': enhan_stoi,
                            'db_sum_noisy_ncm': noisy_ncm, 'db_sum_enhan_ncm': enhan_ncm,
                            'db_sum_voc_noisy_pesq': vocoded_noisy_pesq, 'db_sum_voc_enhan_pesq': vocoded_enhan_pesq,
                            'db_sum_voc_noisy_stoi': vocoded_noisy_stoi, 'db_sum_voc_enhan_stoi': vocoded_enhan_stoi,
                            'db_sum_voc_noisy_ncm': vocoded_noisy_ncm, 'db_sum_voc_enhan_ncm': vocoded_enhan_ncm}

        if noisetype in db_noisetype_score:
            db_noisetype_score[noisetype]['noisetype_count'] += 1
            db_noisetype_score[noisetype]['noisetype_sum_noisy_pesq'] += noisy_pesq
            db_noisetype_score[noisetype]['noisetype_sum_enhan_pesq'] += enhan_pesq
            db_noisetype_score[noisetype]['noisetype_sum_noisy_stoi'] += noisy_stoi
            db_noisetype_score[noisetype]['noisetype_sum_enhan_stoi'] += enhan_stoi
            db_noisetype_score[noisetype]['noisetype_sum_noisy_ncm'] += noisy_ncm
            db_noisetype_score[noisetype]['noisetype_sum_enhan_ncm'] += enhan_ncm
            db_noisetype_score[noisetype]['noisetype_sum_voc_noisy_pesq'] += vocoded_noisy_pesq
            db_noisetype_score[noisetype]['noisetype_sum_voc_enhan_pesq'] += vocoded_enhan_pesq
            db_noisetype_score[noisetype]['noisetype_sum_voc_noisy_stoi'] += vocoded_noisy_stoi
            db_noisetype_score[noisetype]['noisetype_sum_voc_enhan_stoi'] += vocoded_enhan_stoi
            db_noisetype_score[noisetype]['noisetype_sum_voc_noisy_ncm'] += vocoded_noisy_ncm
            db_noisetype_score[noisetype]['noisetype_sum_voc_enhan_ncm'] += vocoded_enhan_ncm

            if db_noisetype in db_noisetype_score[noisetype]:
                # print('update ' + db)
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_count'] += 1
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_noisy_pesq'] += noisy_pesq
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_enhan_pesq'] += enhan_pesq
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_noisy_stoi'] += noisy_stoi
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_enhan_stoi'] += enhan_stoi
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_noisy_ncm'] += noisy_ncm
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_enhan_ncm'] += enhan_ncm
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_voc_noisy_pesq'] += vocoded_noisy_pesq
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_voc_enhan_pesq'] += vocoded_enhan_pesq
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_voc_noisy_stoi'] += vocoded_noisy_stoi
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_voc_enhan_stoi'] += vocoded_enhan_stoi
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_voc_noisy_ncm'] += vocoded_noisy_ncm
                db_noisetype_score[noisetype][db_noisetype]['db_noisetype_sum_voc_enhan_ncm'] += vocoded_enhan_ncm
                # print('db_noisetype =', db_noisetype)
                # print('count =', db_noisetype_score[noisetype][db_noisetype]['db_noisetype_count'])
            else:
                # print('add ' + db_noisetype)
                db_noisetype_order.append(db_noisetype)
                db_noisetype_score[noisetype][db_noisetype] = {'db_noisetype_count': 1,
                                                               'db_noisetype_sum_noisy_pesq': noisy_pesq,
                                                               'db_noisetype_sum_enhan_pesq': enhan_pesq,
                                                               'db_noisetype_sum_noisy_stoi': noisy_stoi,
                                                               'db_noisetype_sum_enhan_stoi': enhan_stoi,
                                                               'db_noisetype_sum_noisy_ncm': noisy_ncm,
                                                               'db_noisetype_sum_enhan_ncm': enhan_ncm,
                                                               'db_noisetype_sum_voc_noisy_pesq': vocoded_noisy_pesq,
                                                               'db_noisetype_sum_voc_enhan_pesq': vocoded_enhan_pesq,
                                                               'db_noisetype_sum_voc_noisy_stoi': vocoded_noisy_stoi,
                                                               'db_noisetype_sum_voc_enhan_stoi': vocoded_enhan_stoi,
                                                               'db_noisetype_sum_voc_noisy_ncm': vocoded_noisy_ncm,
                                                               'db_noisetype_sum_voc_enhan_ncm': vocoded_enhan_ncm}
                # print('db_noisetype =', db_noisetype)
                # print('count =', db_noisetype_score[noisetype][db_noisetype]['db_noisetype_count'])
        else:
            noisetype_order.append(noisetype)
            db_noisetype_score[noisetype] = {'noisetype_count': 1,
                                             'noisetype_sum_noisy_pesq': noisy_pesq,
                                             'noisetype_sum_enhan_pesq': enhan_pesq,
                                             'noisetype_sum_noisy_stoi': noisy_stoi,
                                             'noisetype_sum_enhan_stoi': enhan_stoi,
                                             'noisetype_sum_noisy_ncm': noisy_ncm,
                                             'noisetype_sum_enhan_ncm': enhan_ncm,
                                             'noisetype_sum_voc_noisy_pesq': vocoded_noisy_pesq,
                                             'noisetype_sum_voc_enhan_pesq': vocoded_enhan_pesq,
                                             'noisetype_sum_voc_noisy_stoi': vocoded_noisy_stoi,
                                             'noisetype_sum_voc_enhan_stoi': vocoded_enhan_stoi,
                                             'noisetype_sum_voc_noisy_ncm': vocoded_noisy_ncm,
                                             'noisetype_sum_voc_enhan_ncm': vocoded_enhan_ncm}
            db_noisetype_order.append(db_noisetype)
            db_noisetype_score[noisetype][db_noisetype] = {'db_noisetype_count': 1,
                                                           'db_noisetype_sum_noisy_pesq': noisy_pesq,
                                                           'db_noisetype_sum_enhan_pesq': enhan_pesq,
                                                           'db_noisetype_sum_noisy_stoi': noisy_stoi,
                                                           'db_noisetype_sum_enhan_stoi': enhan_stoi,
                                                           'db_noisetype_sum_noisy_ncm': noisy_ncm,
                                                           'db_noisetype_sum_enhan_ncm': enhan_ncm,
                                                           'db_noisetype_sum_voc_noisy_pesq': vocoded_noisy_pesq,
                                                           'db_noisetype_sum_voc_enhan_pesq': vocoded_enhan_pesq,
                                                           'db_noisetype_sum_voc_noisy_stoi': vocoded_noisy_stoi,
                                                           'db_noisetype_sum_voc_enhan_stoi': vocoded_enhan_stoi,
                                                           'db_noisetype_sum_voc_noisy_ncm': vocoded_noisy_ncm,
                                                           'db_noisetype_sum_voc_enhan_ncm': vocoded_enhan_ncm}
            # print('db_noisetype =', db_noisetype)
            # print('count =', db_noisetype_score[noisetype][db_noisetype]['db_noisetype_count'])

    # print(db_noisetype_score)
    print()
    w.writerow(())

    mean_noisy_pesq = sum_noisy_pesq / count
    mean_enhan_pesq = sum_enhan_pesq / count
    mean_noisy_stoi = sum_noisy_stoi / count
    mean_enhan_stoi = sum_enhan_stoi / count
    mean_noisy_ncm = sum_noisy_ncm / count
    mean_enhan_ncm = sum_enhan_ncm / count
    mean_voc_noisy_pesq = sum_voc_noisy_pesq / count
    mean_voc_enhan_pesq = sum_voc_enhan_pesq / count
    mean_voc_noisy_stoi = sum_voc_noisy_stoi / count
    mean_voc_enhan_stoi = sum_voc_enhan_stoi / count
    mean_voc_noisy_ncm = sum_voc_noisy_ncm / count
    mean_voc_enhan_ncm = sum_voc_enhan_ncm / count

    print('    mean_noisy_pesq = %5.3f,     mean_noisy_stoi = %5.3f,     mean_noisy_ncm = %5.3f' % (mean_noisy_pesq, mean_noisy_stoi, mean_noisy_ncm))
    print('    mean_enhan_pesq = %5.3f,     mean_enhan_stoi = %5.3f,     mean_enhan_ncm = %5.3f' % (mean_enhan_pesq, mean_enhan_stoi, mean_enhan_ncm))
    print()
    print('mean_voc_noisy_pesq = %5.3f, mean_voc_noisy_stoi = %5.3f, mean_voc_noisy_ncm = %5.3f' % (mean_voc_noisy_pesq, mean_voc_noisy_stoi, mean_voc_noisy_ncm))
    print('mean_voc_enhan_pesq = %5.3f, mean_voc_enhan_stoi = %5.3f, mean_voc_enhan_ncm = %5.3f' % (mean_voc_enhan_pesq, mean_voc_enhan_stoi, mean_voc_enhan_ncm))
    print()

    for db_noisetype in db_noisetype_order:
        # print('db_noisetype =', db_noisetype)
        noisetype = db_noisetype.rsplit('_', 1)[-2]
        value = db_noisetype_score[noisetype][db_noisetype]
        db_noisetype_mean_noisy_pesq = value['db_noisetype_sum_noisy_pesq'] / value['db_noisetype_count']
        db_noisetype_mean_enhan_pesq = value['db_noisetype_sum_enhan_pesq'] / value['db_noisetype_count']
        db_noisetype_mean_noisy_stoi = value['db_noisetype_sum_noisy_stoi'] / value['db_noisetype_count']
        db_noisetype_mean_enhan_stoi = value['db_noisetype_sum_enhan_stoi'] / value['db_noisetype_count']
        db_noisetype_mean_noisy_ncm = value['db_noisetype_sum_noisy_ncm'] / value['db_noisetype_count']
        db_noisetype_mean_enhan_ncm = value['db_noisetype_sum_enhan_ncm'] / value['db_noisetype_count']
        db_noisetype_mean_voc_noisy_pesq = value['db_noisetype_sum_voc_noisy_pesq'] / value['db_noisetype_count']
        db_noisetype_mean_voc_enhan_pesq = value['db_noisetype_sum_voc_enhan_pesq'] / value['db_noisetype_count']
        db_noisetype_mean_voc_noisy_stoi = value['db_noisetype_sum_voc_noisy_stoi'] / value['db_noisetype_count']
        db_noisetype_mean_voc_enhan_stoi = value['db_noisetype_sum_voc_enhan_stoi'] / value['db_noisetype_count']
        db_noisetype_mean_voc_noisy_ncm = value['db_noisetype_sum_voc_noisy_ncm'] / value['db_noisetype_count']
        db_noisetype_mean_voc_enhan_ncm = value['db_noisetype_sum_voc_enhan_ncm'] / value['db_noisetype_count']

        # save means for visdom
        db_noisetype_score[noisetype][db_noisetype] = {'db_noisetype_mean_noisy_pesq': db_noisetype_mean_noisy_pesq,
                                                       'db_noisetype_mean_enhan_pesq': db_noisetype_mean_enhan_pesq,
                                                       'db_noisetype_mean_noisy_stoi': db_noisetype_mean_noisy_stoi,
                                                       'db_noisetype_mean_enhan_stoi': db_noisetype_mean_enhan_stoi,
                                                       'db_noisetype_mean_noisy_ncm': db_noisetype_mean_noisy_ncm,
                                                       'db_noisetype_mean_enhan_ncm': db_noisetype_mean_enhan_ncm,
                                                       'db_noisetype_mean_voc_noisy_pesq': db_noisetype_mean_voc_noisy_pesq,
                                                       'db_noisetype_mean_voc_enhan_pesq': db_noisetype_mean_voc_enhan_pesq,
                                                       'db_noisetype_mean_voc_noisy_stoi': db_noisetype_mean_voc_noisy_stoi,
                                                       'db_noisetype_mean_voc_enhan_stoi': db_noisetype_mean_voc_enhan_stoi,
                                                       'db_noisetype_mean_voc_noisy_ncm': db_noisetype_mean_voc_noisy_ncm,
                                                       'db_noisetype_mean_voc_enhan_ncm': db_noisetype_mean_voc_enhan_ncm}

        w.writerow((db_noisetype + ' mean', db_noisetype_mean_noisy_pesq, db_noisetype_mean_enhan_pesq,
                                            db_noisetype_mean_noisy_stoi, db_noisetype_mean_enhan_stoi,
                                            db_noisetype_mean_noisy_ncm, db_noisetype_mean_enhan_ncm,
                                            db_noisetype_mean_voc_noisy_pesq, db_noisetype_mean_voc_enhan_pesq,
                                            db_noisetype_mean_voc_noisy_stoi, db_noisetype_mean_voc_enhan_stoi,
                                            db_noisetype_mean_voc_noisy_ncm, db_noisetype_mean_voc_enhan_ncm))

    # print(db_noisetype_score)
    w.writerow(())

    visdom_noisy_pesq = []
    visdom_enhan_pesq = []
    visdom_noisy_stoi = []
    visdom_enhan_stoi = []
    visdom_noisy_ncm = []
    visdom_enhan_ncm = []
    visdom_voc_noisy_pesq = []
    visdom_voc_enhan_pesq = []
    visdom_voc_noisy_stoi = []
    visdom_voc_enhan_stoi = []
    visdom_voc_noisy_ncm = []
    visdom_voc_enhan_ncm = []

    for noisetype in noisetype_order:
        dictionary = db_noisetype_score[noisetype]
        noisetype_noisy_pesq = []
        noisetype_enhan_pesq = []
        noisetype_noisy_stoi = []
        noisetype_enhan_stoi = []
        noisetype_noisy_ncm = []
        noisetype_enhan_ncm = []
        noisetype_voc_noisy_pesq = []
        noisetype_voc_enhan_pesq = []
        noisetype_voc_noisy_stoi = []
        noisetype_voc_enhan_stoi = []
        noisetype_voc_noisy_ncm = []
        noisetype_voc_enhan_ncm = []

        for db in db_order:
            db_noisetype = noisetype + '_' + db
            # print('db_noisetype =', db_noisetype)
            value = dictionary[db_noisetype]
            db_noisetype_mean_noisy_pesq = value['db_noisetype_mean_noisy_pesq']
            db_noisetype_mean_enhan_pesq = value['db_noisetype_mean_enhan_pesq']
            db_noisetype_mean_noisy_stoi = value['db_noisetype_mean_noisy_stoi']
            db_noisetype_mean_enhan_stoi = value['db_noisetype_mean_enhan_stoi']
            db_noisetype_mean_noisy_ncm = value['db_noisetype_mean_noisy_ncm']
            db_noisetype_mean_enhan_ncm = value['db_noisetype_mean_enhan_ncm']
            db_noisetype_mean_voc_noisy_pesq = value['db_noisetype_mean_voc_noisy_pesq']
            db_noisetype_mean_voc_enhan_pesq = value['db_noisetype_mean_voc_enhan_pesq']
            db_noisetype_mean_voc_noisy_stoi = value['db_noisetype_mean_voc_noisy_stoi']
            db_noisetype_mean_voc_enhan_stoi = value['db_noisetype_mean_voc_enhan_stoi']
            db_noisetype_mean_voc_noisy_ncm = value['db_noisetype_mean_voc_noisy_ncm']
            db_noisetype_mean_voc_enhan_ncm = value['db_noisetype_mean_voc_enhan_ncm']
            noisetype_noisy_pesq.append(db_noisetype_mean_noisy_pesq)
            noisetype_enhan_pesq.append(db_noisetype_mean_enhan_pesq)
            noisetype_noisy_stoi.append(db_noisetype_mean_noisy_stoi)
            noisetype_enhan_stoi.append(db_noisetype_mean_enhan_stoi)
            noisetype_noisy_ncm.append(db_noisetype_mean_noisy_ncm)
            noisetype_enhan_ncm.append(db_noisetype_mean_enhan_ncm)
            noisetype_voc_noisy_pesq.append(db_noisetype_mean_voc_noisy_pesq)
            noisetype_voc_enhan_pesq.append(db_noisetype_mean_voc_enhan_pesq)
            noisetype_voc_noisy_stoi.append(db_noisetype_mean_voc_noisy_stoi)
            noisetype_voc_enhan_stoi.append(db_noisetype_mean_voc_enhan_stoi)
            noisetype_voc_noisy_ncm.append(db_noisetype_mean_voc_noisy_ncm)
            noisetype_voc_enhan_ncm.append(db_noisetype_mean_voc_enhan_ncm)

        visdom_noisy_pesq.append(noisetype_noisy_pesq)
        visdom_enhan_pesq.append(noisetype_enhan_pesq)
        visdom_noisy_stoi.append(noisetype_noisy_stoi)
        visdom_enhan_stoi.append(noisetype_enhan_stoi)
        visdom_noisy_ncm.append(noisetype_noisy_ncm)
        visdom_enhan_ncm.append(noisetype_enhan_ncm)
        visdom_voc_noisy_pesq.append(noisetype_voc_noisy_pesq)
        visdom_voc_enhan_pesq.append(noisetype_voc_enhan_pesq)
        visdom_voc_noisy_stoi.append(noisetype_voc_noisy_stoi)
        visdom_voc_enhan_stoi.append(noisetype_voc_enhan_stoi)
        visdom_voc_noisy_ncm.append(noisetype_voc_noisy_ncm)
        visdom_voc_enhan_ncm.append(noisetype_voc_enhan_ncm)

    visdom_noisetype_noisy_pesq = []
    visdom_noisetype_enhan_pesq = []
    visdom_noisetype_noisy_stoi = []
    visdom_noisetype_enhan_stoi = []
    visdom_noisetype_noisy_ncm = []
    visdom_noisetype_enhan_ncm = []
    visdom_noisetype_voc_noisy_pesq = []
    visdom_noisetype_voc_enhan_pesq = []
    visdom_noisetype_voc_noisy_stoi = []
    visdom_noisetype_voc_enhan_stoi = []
    visdom_noisetype_voc_noisy_ncm = []
    visdom_noisetype_voc_enhan_ncm = []

    for noisetype in noisetype_order:
        value = db_noisetype_score[noisetype]
        noisetype_mean_noisy_pesq = value['noisetype_sum_noisy_pesq'] / value['noisetype_count']
        noisetype_mean_enhan_pesq = value['noisetype_sum_enhan_pesq'] / value['noisetype_count']
        noisetype_mean_noisy_stoi = value['noisetype_sum_noisy_stoi'] / value['noisetype_count']
        noisetype_mean_enhan_stoi = value['noisetype_sum_enhan_stoi'] / value['noisetype_count']
        noisetype_mean_noisy_ncm = value['noisetype_sum_noisy_ncm'] / value['noisetype_count']
        noisetype_mean_enhan_ncm = value['noisetype_sum_enhan_ncm'] / value['noisetype_count']
        noisetype_mean_voc_noisy_pesq = value['noisetype_sum_voc_noisy_pesq'] / value['noisetype_count']
        noisetype_mean_voc_enhan_pesq = value['noisetype_sum_voc_enhan_pesq'] / value['noisetype_count']
        noisetype_mean_voc_noisy_stoi = value['noisetype_sum_voc_noisy_stoi'] / value['noisetype_count']
        noisetype_mean_voc_enhan_stoi = value['noisetype_sum_voc_enhan_stoi'] / value['noisetype_count']
        noisetype_mean_voc_noisy_ncm = value['noisetype_sum_voc_noisy_ncm'] / value['noisetype_count']
        noisetype_mean_voc_enhan_ncm = value['noisetype_sum_voc_enhan_ncm'] / value['noisetype_count']

        visdom_noisetype_noisy_pesq.append(noisetype_mean_noisy_pesq)
        visdom_noisetype_enhan_pesq.append(noisetype_mean_enhan_pesq)
        visdom_noisetype_noisy_stoi.append(noisetype_mean_noisy_stoi)
        visdom_noisetype_enhan_stoi.append(noisetype_mean_enhan_stoi)
        visdom_noisetype_noisy_ncm.append(noisetype_mean_noisy_ncm)
        visdom_noisetype_enhan_ncm.append(noisetype_mean_enhan_ncm)
        visdom_noisetype_voc_noisy_pesq.append(noisetype_mean_voc_noisy_pesq)
        visdom_noisetype_voc_enhan_pesq.append(noisetype_mean_voc_enhan_pesq)
        visdom_noisetype_voc_noisy_stoi.append(noisetype_mean_voc_noisy_stoi)
        visdom_noisetype_voc_enhan_stoi.append(noisetype_mean_voc_enhan_stoi)
        visdom_noisetype_voc_noisy_ncm.append(noisetype_mean_voc_noisy_ncm)
        visdom_noisetype_voc_enhan_ncm.append(noisetype_mean_voc_enhan_ncm)

        w.writerow((noisetype + ' mean', noisetype_mean_noisy_pesq, noisetype_mean_enhan_pesq,
                                         noisetype_mean_noisy_stoi, noisetype_mean_enhan_stoi,
                                         noisetype_mean_noisy_ncm, noisetype_mean_enhan_ncm,
                                         noisetype_mean_voc_noisy_pesq, noisetype_mean_voc_enhan_pesq,
                                         noisetype_mean_voc_noisy_stoi, noisetype_mean_voc_enhan_stoi,
                                         noisetype_mean_voc_noisy_ncm, noisetype_mean_voc_enhan_ncm))

    w.writerow(())

    visdom_db_noisy_pesq = []
    visdom_db_enhan_pesq = []
    visdom_db_noisy_stoi = []
    visdom_db_enhan_stoi = []
    visdom_db_noisy_ncm = []
    visdom_db_enhan_ncm = []
    visdom_db_voc_noisy_pesq = []
    visdom_db_voc_enhan_pesq = []
    visdom_db_voc_noisy_stoi = []
    visdom_db_voc_enhan_stoi = []
    visdom_db_voc_noisy_ncm = []
    visdom_db_voc_enhan_ncm = []

    for db in db_order:
        value = db_score[db]
        db_mean_noisy_pesq = value['db_sum_noisy_pesq'] / value['db_count']
        db_mean_enhan_pesq = value['db_sum_enhan_pesq'] / value['db_count']
        db_mean_noisy_stoi = value['db_sum_noisy_stoi'] / value['db_count']
        db_mean_enhan_stoi = value['db_sum_enhan_stoi'] / value['db_count']
        db_mean_noisy_ncm = value['db_sum_noisy_ncm'] / value['db_count']
        db_mean_enhan_ncm = value['db_sum_enhan_ncm'] / value['db_count']
        db_mean_voc_noisy_pesq = value['db_sum_voc_noisy_pesq'] / value['db_count']
        db_mean_voc_enhan_pesq = value['db_sum_voc_enhan_pesq'] / value['db_count']
        db_mean_voc_noisy_stoi = value['db_sum_voc_noisy_stoi'] / value['db_count']
        db_mean_voc_enhan_stoi = value['db_sum_voc_enhan_stoi'] / value['db_count']
        db_mean_voc_noisy_ncm = value['db_sum_voc_noisy_ncm'] / value['db_count']
        db_mean_voc_enhan_ncm = value['db_sum_voc_enhan_ncm'] / value['db_count']

        visdom_db_noisy_pesq.append(db_mean_noisy_pesq)
        visdom_db_enhan_pesq.append(db_mean_enhan_pesq)
        visdom_db_noisy_stoi.append(db_mean_noisy_stoi)
        visdom_db_enhan_stoi.append(db_mean_enhan_stoi)
        visdom_db_noisy_ncm.append(db_mean_noisy_ncm)
        visdom_db_enhan_ncm.append(db_mean_enhan_ncm)
        visdom_db_voc_noisy_pesq.append(db_mean_voc_noisy_pesq)
        visdom_db_voc_enhan_pesq.append(db_mean_voc_enhan_pesq)
        visdom_db_voc_noisy_stoi.append(db_mean_voc_noisy_stoi)
        visdom_db_voc_enhan_stoi.append(db_mean_voc_enhan_stoi)
        visdom_db_voc_noisy_ncm.append(db_mean_voc_noisy_ncm)
        visdom_db_voc_enhan_ncm.append(db_mean_voc_enhan_ncm)

        w.writerow((db + ' mean', db_mean_noisy_pesq, db_mean_enhan_pesq,
                                  db_mean_noisy_stoi, db_mean_enhan_stoi,
                                  db_mean_noisy_ncm, db_mean_enhan_ncm,
                                  db_mean_voc_noisy_pesq, db_mean_voc_enhan_pesq,
                                  db_mean_voc_noisy_stoi, db_mean_voc_enhan_stoi,
                                  db_mean_voc_noisy_ncm, db_mean_voc_enhan_ncm))

    w.writerow(())
    w.writerow(('total mean', mean_noisy_pesq, mean_enhan_pesq,
                              mean_noisy_stoi, mean_enhan_stoi,
                              mean_noisy_ncm, mean_enhan_ncm,
                              mean_voc_noisy_pesq, mean_voc_enhan_pesq,
                              mean_voc_noisy_stoi, mean_voc_enhan_stoi,
                              mean_voc_noisy_ncm, mean_voc_enhan_ncm))
    f.close()

    # remove the files created during scoring PESQ
    subprocess.call('rm cln_%s.wav' % model_detail_time, shell=True)
    subprocess.call('rm nsy_%s.wav' % model_detail_time, shell=True)
    subprocess.call('rm enh_%s.wav' % model_detail_time, shell=True)

    # remove the by-product created by the PESQ execute file
    subprocess.call(['rm', '_pesq_itu_results.txt'])
    subprocess.call(['rm', '_pesq_results.txt'])

    print('Visdom activating.')
    viz = Visdom(env='%s' % model_detail)

    ### visdom overall
    # PESQ
    viz.bar(
        visdom_noisy_pesq, win='score_noisy_pesq',
        opts=dict(title='Noisy PESQ overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=3, ytickmin=0)
    )
    viz.bar(
        visdom_enhan_pesq, win='score_enhan_pesq',
        opts=dict(title='Enhanced PESQ overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=3, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_enhan_pesq) - np.array(visdom_noisy_pesq)), win='improve_pesq',
        opts=dict(title='PESQ Improvement overall', xlabel='Noise Type', ylabel='Enhanced - Noisy', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False)
    )
    viz.bar(
        visdom_voc_noisy_pesq, win='score_voc_noisy_pesq',
        opts=dict(title='Voc Noisy PESQ overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=3, ytickmin=0)
    )
    viz.bar(
        visdom_voc_enhan_pesq, win='score_voc_enhan_pesq',
        opts=dict(title='Voc Enhanced PESQ overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=3, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_voc_enhan_pesq) - np.array(visdom_voc_noisy_pesq)), win='voc_improve_pesq',
        opts=dict(title='Voc PESQ Improvement overall', xlabel='Noise Type', ylabel='Voc Enhanced - Voc Noisy', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False)
    )

    # STOI
    viz.bar(
        visdom_noisy_stoi, win='score_noisy_stoi',
        opts=dict(title='Noisy STOI overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        visdom_enhan_stoi, win='score_enhan_stoi',
        opts=dict(title='Enhanced STOI overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_enhan_stoi) - np.array(visdom_noisy_stoi)), win='improve_stoi',
        opts=dict(title='STOI Improvement overall', xlabel='Noise Type', ylabel='Enhanced - Noisy', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False)
    )
    viz.bar(
        visdom_voc_noisy_stoi, win='score_voc_noisy_stoi',
        opts=dict(title='Voc Noisy STOI overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        visdom_voc_enhan_stoi, win='score_voc_enhan_stoi',
        opts=dict(title='Voc Enhanced STOI overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_voc_enhan_stoi) - np.array(visdom_voc_noisy_stoi)), win='voc_improve_stoi',
        opts=dict(title='Voc STOI Improvement overall', xlabel='Noise Type', ylabel='Voc Enhanced - Voc Noisy', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False)
    )

    # NCM
    viz.bar(
        visdom_noisy_ncm, win='score_noisy_ncm',
        opts=dict(title='Noisy NCM overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        visdom_enhan_ncm, win='score_enhan_ncm',
        opts=dict(title='Enhanced NCM overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_enhan_ncm) - np.array(visdom_noisy_ncm)), win='improve_ncm',
        opts=dict(title='NCM Improvement overall', xlabel='Noise Type', ylabel='Enhanced - Noisy', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False)
    )
    viz.bar(
        visdom_voc_noisy_ncm, win='score_voc_noisy_ncm',
        opts=dict(title='Voc Noisy NCM overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        visdom_voc_enhan_ncm, win='score_voc_enhan_ncm',
        opts=dict(title='Voc Enhanced NCM overall', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_voc_enhan_ncm) - np.array(visdom_voc_noisy_ncm)), win='voc_improve_ncm',
        opts=dict(title='Voc NCM Improvement overall', xlabel='Noise Type', ylabel='Voc Enhanced - Voc Noisy', showlegend=True,
                  rownames=noisetype_order, legend=db_order, stacked=False)
    )

    ### visdom by noise type
    # PESQ
    viz.bar(
        np.column_stack((visdom_noisetype_noisy_pesq, visdom_noisetype_enhan_pesq)), win='score_noisetype_pesq',
        opts=dict(title='PESQ by Noise Type', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=['Noisy', 'Enhanced'], stacked=False, ytickmax=3, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_noisetype_enhan_pesq) - np.array(visdom_noisetype_noisy_pesq)), win='improve_noisetype_pesq',
        opts=dict(title='PESQ Improvement by Noise Type', xlabel='Noise Type', ylabel='Enhanced - Noisy',
                  rownames=noisetype_order, stacked=False)
    )
    viz.bar(
        np.column_stack((visdom_noisetype_voc_noisy_pesq, visdom_noisetype_voc_enhan_pesq)), win='voc_score_noisetype_pesq',
        opts=dict(title='Voc PESQ by Noise Type', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=['Voc Noisy', 'Voc Enhanced'], stacked=False, ytickmax=3, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_noisetype_voc_enhan_pesq) - np.array(visdom_noisetype_voc_noisy_pesq)), win='voc_improve_noisetype_pesq',
        opts=dict(title='Voc PESQ Improvement by Noise Type', xlabel='Noise Type', ylabel='Voc Enhanced - Voc Noisy',
                  rownames=noisetype_order, stacked=False)
    )

    # STOI
    viz.bar(
        np.column_stack((visdom_noisetype_noisy_stoi, visdom_noisetype_enhan_stoi)), win='score_noisetype_stoi',
        opts=dict(title='STOI by Noise Type', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=['Noisy', 'Enhanced'], stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_noisetype_enhan_stoi) - np.array(visdom_noisetype_noisy_stoi)), win='improve_noisetype_stoi',
        opts=dict(title='STOI Improvement by Noise Type', xlabel='Noise Type', ylabel='Enhanced - Noisy',
                  rownames=noisetype_order, stacked=False)
    )
    viz.bar(
        np.column_stack((visdom_noisetype_voc_noisy_stoi, visdom_noisetype_voc_enhan_stoi)), win='voc_score_noisetype_stoi',
        opts=dict(title='Voc STOI by Noise Type', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=['Voc Noisy', 'Voc Enhanced'], stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_noisetype_voc_enhan_stoi) - np.array(visdom_noisetype_voc_noisy_stoi)), win='voc_improve_noisetype_stoi',
        opts=dict(title='Voc STOI Improvement by Noise Type', xlabel='Noise Type', ylabel='Voc Enhanced - Voc Noisy',
                  rownames=noisetype_order, stacked=False)
    )

    # NCM
    viz.bar(
        np.column_stack((visdom_noisetype_noisy_ncm, visdom_noisetype_enhan_ncm)), win='score_noisetype_ncm',
        opts=dict(title='NCM by Noise Type', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=['Noisy', 'Enhanced'], stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_noisetype_enhan_ncm) - np.array(visdom_noisetype_noisy_ncm)), win='improve_noisetype_ncm',
        opts=dict(title='NCM Improvement by Noise Type', xlabel='Noise Type', ylabel='Enhanced - Noisy',
                  rownames=noisetype_order, stacked=False)
    )
    viz.bar(
        np.column_stack((visdom_noisetype_voc_noisy_ncm, visdom_noisetype_voc_enhan_ncm)), win='voc_score_noisetype_ncm',
        opts=dict(title='Voc NCM by Noise Type', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=noisetype_order, legend=['Voc Noisy', 'Voc Enhanced'], stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_noisetype_voc_enhan_ncm) - np.array(visdom_noisetype_voc_noisy_ncm)), win='voc_improve_noisetype_ncm',
        opts=dict(title='Voc NCM Improvement by Noise Type', xlabel='Noise Type', ylabel='Voc Enhanced - Voc Noisy',
                  rownames=noisetype_order, stacked=False)
    )

    ### visdom by SNR
    # PESQ
    viz.bar(
        np.column_stack((visdom_db_noisy_pesq, visdom_db_enhan_pesq)), win='score_db_pesq',
        opts=dict(title='PESQ by SNR', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=db_order, legend=['Noisy', 'Enhanced'], stacked=False, ytickmax=3, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_db_enhan_pesq) - np.array(visdom_db_noisy_pesq)), win='improve_db_pesq',
        opts=dict(title='PESQ Improvement by SNR', xlabel='SNR', ylabel='Enhanced - Noisy', rownames=db_order, stacked=False)
    )
    viz.bar(
        np.column_stack((visdom_db_voc_noisy_pesq, visdom_db_voc_enhan_pesq)), win='voc_score_db_pesq',
        opts=dict(title='Voc PESQ by SNR', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=db_order, legend=['Voc Noisy', 'Voc Enhanced'], stacked=False, ytickmax=3, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_db_voc_enhan_pesq) - np.array(visdom_db_voc_noisy_pesq)), win='voc_improve_db_pesq',
        opts=dict(title='Voc PESQ Improvement by SNR', xlabel='SNR', ylabel='Voc Enhanced - Voc Noisy', rownames=db_order, stacked=False)
    )

    # STOI
    viz.bar(
        np.column_stack((visdom_db_noisy_stoi, visdom_db_enhan_stoi)), win='score_db_stoi',
        opts=dict(title='STOI by SNR', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=db_order, legend=['Noisy', 'Enhanced'], stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_db_enhan_stoi) - np.array(visdom_db_noisy_stoi)), win='improve_db_stoi',
        opts=dict(title='STOI Improvement by SNR', xlabel='SNR', ylabel='Enhanced - Noisy', rownames=db_order, stacked=False)
    )
    viz.bar(
        np.column_stack((visdom_db_voc_noisy_stoi, visdom_db_voc_enhan_stoi)), win='voc_score_db_stoi',
        opts=dict(title='Voc STOI by SNR', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=db_order, legend=['Voc Noisy', 'Voc Enhanced'], stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_db_voc_enhan_stoi) - np.array(visdom_db_voc_noisy_stoi)), win='voc_improve_db_stoi',
        opts=dict(title='Voc STOI Improvement by SNR', xlabel='SNR', ylabel='Voc Enhanced - Voc Noisy', rownames=db_order, stacked=False)
    )

    # NCM
    viz.bar(
        np.column_stack((visdom_db_noisy_ncm, visdom_db_enhan_ncm)), win='score_db_ncm',
        opts=dict(title='NCM by SNR', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=db_order, legend=['Noisy', 'Enhanced'], stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_db_enhan_ncm) - np.array(visdom_db_noisy_ncm)), win='improve_db_ncm',
        opts=dict(title='NCM Improvement by SNR', xlabel='SNR', ylabel='Enhanced - Noisy', rownames=db_order, stacked=False)
    )
    viz.bar(
        np.column_stack((visdom_db_voc_noisy_ncm, visdom_db_voc_enhan_ncm)), win='voc_score_db_ncm',
        opts=dict(title='Voc NCM by SNR', xlabel='Noise Type', ylabel='Scores', showlegend=True,
                  rownames=db_order, legend=['Voc Noisy', 'Voc Enhanced'], stacked=False, ytickmax=1, ytickmin=0)
    )
    viz.bar(
        (np.array(visdom_db_voc_enhan_ncm) - np.array(visdom_db_voc_noisy_ncm)), win='voc_improve_db_ncm',
        opts=dict(title='Voc NCM Improvement by SNR', xlabel='SNR', ylabel='Voc Enhanced - Voc Noisy', rownames=db_order, stacked=False)
    )
    # save visdom score
    print('Visdom saving.')
    result_visdom_path = result_model_path + 'visdom_score[%s].log' % model_detail
    vis.create_log_at(result_visdom_path, model_detail)
    print()

    end_time = time.time()

    score_time = cal_time(start_time, end_time)
    print('Scoring complete.\n')

    return score_time
