import random
random.seed('supercalifragilisticexpialidocious')

import torch
from torchsummaryX import summary
import os
import argparse
import time
from torch.optim import Adam

from prepare_path_list_AEC import prepare_path_list, dataset, dataset_path
from data_generator_AEC import wav_Dataset
from build_model import *
from utils_AEC import model_detail_string, train, test, cal_time
from scoring_AEC import prepare_scoring_list, write_score

parser = argparse.ArgumentParser()

########## training ##########
parser.add_argument('--retrain', action='store_true', help='to train a new model or to retrain an existing model.')
parser.add_argument('--model', type=str, default='AEC_BSRU', help='options: AEC_BSRU')
parser.add_argument('--loss', type=str, default='MSE', help='option: MSE')
parser.add_argument('--opt', type=str, default='Adam', help='option: Adam')
parser.add_argument('--epochs', type=int, default=500, help='the last epoch wanted to be trained.')
parser.add_argument('--train_batch_size', type=int, default=1, help='the batch size wanted to be trained.')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate of the optimizer.')
########## testing ##########
parser.add_argument('--retest', action='store_true', help='to test or retest an existing model.')
parser.add_argument('--test_batch_size', type=int, default=1, help='the batch size wanted to be tested.')
########## scoring ##########
parser.add_argument('--rescore', action='store_true', help='to rescore test wavs. scoring will automatically start after testing even if this argument is not triggered.')

args = parser.parse_args()

########## training ##########
retrain = args.retrain
model_name = args.model
loss_name = args.loss
opt_name = args.opt
epochs = args.epochs
train_batch_size = args.train_batch_size
lr = args.learning_rate
########## testing ##########
retest = args.retest
test_batch_size = args.test_batch_size
########## scoring ##########
rescore = args.rescore

if __name__ == '__main__':

    # ********** starting **********
    print('\n********** starting **********\n')

    print('This code is for the %s model with the dataset of %s.\n' % (model_name, dataset))

    start_time = time.time()

    # ********** check cuda **********

    gpu_amount = torch.cuda.device_count()

    print('#################################')
    print('torch.cuda.is_available() =', torch.cuda.is_available())
    # torch.cuda.is_available() reveals if any gpu can be used and if any gpu is assigned with CUDA_VISIBLE_DEVICES in bash script
    print('torch.cuda.device_count() =', gpu_amount)
    # torch.cuda.device_count() reveals how many gpus can be used, which is implicitly assigned in bash script with CUDA_VISIBLE_DEVICES
    # in bash script, CUDA_VISIBLE_DEVICES is assigned by the specific numbers of gpus, but not the amount
    print('#################################\n')

    device = os.environ['CUDA_VISIBLE_DEVICES']
    device = 'cpu' if device == '' else 'cuda'
    # setting default to 'cuda' indicates the program will use gpu:0
    # 0 here means the first gpu number assigned by CUDA_VISIBLE_DEVICES, but not the real gpu:0
    device = torch.device(device)
    # print('device =', device)

    # ********** model declare **********

    if model_name == 'AEC_BSRU':
        model = AEC_BSRU().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_BSRU4':
        model = AEC_BSRU4().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_BSRU8':
        model = AEC_BSRU8().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_BSRU20':
        model = AEC_BSRU20().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_CSRU':
        model = AEC_CSRU().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_CSRU3':
        model = AEC_CSRU3().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_CSRU3_TANH':
        model = AEC_CSRU3_TANH().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_CSRU4':
        model = AEC_CSRU4().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_CSRU3_3':
        model = AEC_CSRU3_3().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_CSRU3_5':
        model = AEC_CSRU3_5().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_CSRU3_7':
        model = AEC_CSRU3_7().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    elif model_name == 'AEC_CSRU3_9':
        model = AEC_CSRU3_9().to(device)
        input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram
    # elif model_name == 'NEW_MODEL':
    #     model = NEW_MODEL().to(device)
    #     input_size = [train_batch_size, 257, 150] # 150 is just an example for frame length of spectrogram

    ####################################### My First AEC Model (start of code editing) #########################################
    ############################################################################################################################
    elif model_name == 'First_AEC_model_4':
        model = First_AEC_model_4().to(device)
        input_size = [train_batch_size, 257, 150]   # original: 150 

    ################################################### (end of code editing) ##################################################
    ############################################################################################################################

    else:
        raise NameError('custom models (with the parameters of input_size) should be written in main_AEC.py,\n'
             '           and the structure of the models should be written in build_model.py.')

    # criterion
    if loss_name == 'MSE':
        criterion = torch.nn.MSELoss()
    # elif loss_name == 'NEW_LOSS':
    #     criterion = NEW_LOSS()
    else:
        raise NameError('loss undefined.')

    # optimizer
    if opt_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    # elif opt_name == 'NEW_OPT':
    #     optimizer = NEW_OPT(model.parameters(), lr=lr)
    else:
        raise NameError('optimizer undefined.')

    # ********** result path **********

    result_root_path = '../result/'
    result_dir_path = result_root_path + '%s/' % dataset

    model_detail = model_detail_string(model_name, epochs, lr, train_batch_size)

    result_model_path = result_dir_path + model_detail + '/'
    result_audio_path = result_model_path + 'audio/'

    # ********** init **********

    if epochs <= 0:
        raise ValueError('epochs should be larger than zero.')

    if retrain:
        model.apply(weights_init)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1) # lr fixed
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5,10], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)

    train_path_list, val_path_list, test_path_list = prepare_path_list()

    # ********** training **********

    if retrain:
        print('\n********** training **********\n')

        train_dataset = wav_Dataset(data_path_list=train_path_list, mode='training')
        val_dataset = wav_Dataset(data_path_list=val_path_list, mode='validation')

        if not os.path.exists(result_model_path):
            os.makedirs(result_model_path)

        # with torch.autograd.profiler.profile() as prof:
        train_time = train(device, model, train_dataset, val_dataset, epochs, train_batch_size,
                           criterion, optimizer, scheduler, result_model_path)
        # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
        # exit()
        torch.cuda.empty_cache()

    if retrain or retest:
        audio_input_size = torch.zeros(*input_size).to(device)
        summary(model, audio_input_size)

        print()

    # exit() # for training debug

    # ********** testing **********

    if retest:
        print('\n********** testing **********\n')

        checkpoint = torch.load(result_model_path + 'best_model[%s].tar' % model_detail, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_dataset = wav_Dataset(data_path_list=test_path_list, mode='testing')

        if not os.path.exists(result_audio_path):
            os.makedirs(result_audio_path)

        # with torch.autograd.profiler.profile() as prof:
        test_time = test(device, model, test_dataset, test_batch_size, result_audio_path)
        # print(prof.key_averages().table(sort_by='self_cpu_time_total'))
        # exit()
        torch.cuda.empty_cache()

    # ********** scoring **********

    if rescore:
        print('\n********** scoring **********\n')

        path_list = prepare_scoring_list(dataset_path, result_audio_path)
        score_time = write_score(path_list, result_model_path)

    # ********** ending **********

    print('\n********** ending **********\n')

    end_time = time.time()
    code_time = cal_time(start_time, end_time)

    print('This code is for the %s model with the dataset of %s.\n' % (model_name, dataset))

    if retrain:
        print('      Trained for %2d day %2d hr %2d min.' % train_time)

    if retest:
        print('       Tested for %2d day %2d hr %2d min.' % test_time)

    if rescore:
        print('       Scored for %2d day %2d hr %2d min.' % score_time)

    print('This code ran for %2d day %2d hr %2d min.\n' % code_time)
