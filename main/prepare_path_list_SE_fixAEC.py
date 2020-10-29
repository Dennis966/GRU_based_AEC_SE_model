# change speakers, noise type and train/val/test directory of AV_dataset

### speakers
# M: 01, 02, 03, 04, 05, 06, 08, 09, 14, 15, 16, 17, 18
# F: 07, 10, 11, 12, 13

### noise type
# n1-n17: Crowd noise
# n18-n29: Machine noise
# n30-n43: Alarm and siren
# n44-n46: Traffic and car noise
# n47-n55: Animal sound
# n56-n69: Water sound
# n70-n78: Wind
# n79-n82: Bell
# n83-n85: Cough
# n86: Clap
# n87: Snore
# n88: Click
# n88-n90: Laugh
# n91-n92: Yawn
# n93: Cry
# n94: Shower
# n95: Tooth brushing
# n96-n97: Footsteps
# n98: Door moving
# n99-n100: Phone dialing

# dataset_path = '/mnt/Nas/Audio_Visual_Corpus/'
dataset_path = '/mnt/md1/user_sychuang/dataset/'

dataset = 'AV_enh'

train_spk_list = [1, 2, 3, 4, 7, 10, 11, 12] # train 4m, 4f
val_spk_list = [1, 2, 3, 4, 7, 10, 11, 12]
test_spk_list = [5, 13] # test 1m, 1f

noise_train_list = ['n' + str(i) for i in range(1, 101)] # noise n1-100
noise_val_list = ['n' + str(i) for i in range(1, 101)] # noise n1-100
noise_test_list = ['babycry', 'engine', 'm2talker', 'music', 'pink', 'street']

snr_train_list = [0, 6, 12]
snr_val_list = [0, 6, 12]
snr_test_list = [0, 5, 10]

def prepare_train_path_list():
    train_spk_path_list = [dataset_path + 'SP' + str(i).zfill(2) + '/' for i in train_spk_list]

    train_path_list = []

    if dataset == 'AV_enh':
        for spk_path in train_spk_path_list:
            for noise in noise_train_list:
                for snr in snr_train_list:
                    noise_snrdb = noise + '_' + str(snr).replace('-', 'n') + 'db'
                    # (clean_train_path, noisy_train_path, clean_ci_train_path)
                    train_path_list.extend([
                    (spk_path + 'audio/clean/train/',
                     spk_path + 'audio/noisy_enh/train/' + noise_snrdb + '/',
                     spk_path + 'audio_voc/clean/train/')
                    ])

    return train_path_list

def prepare_val_path_list():
    train_spk_path_list = [dataset_path + 'SP' + str(i).zfill(2) + '/' for i in val_spk_list]

    val_path_list = []

    if dataset == 'AV_enh':
        for spk_path in train_spk_path_list:
            for noise in noise_val_list:
                for snr in snr_val_list:
                    noise_snrdb = noise + '_' + str(snr).replace('-', 'n') + 'db'
                    # (clean_val_path, noisy_val_path, clean_ci_val_path)
                    val_path_list.extend([
                    (spk_path + 'audio/clean/val/',
                     spk_path + 'audio/noisy_enh/val/' + noise_snrdb + '/',
                     spk_path + 'audio_voc/clean/val/')
                    ])

    return val_path_list

def prepare_test_path_list():
    test_spk_path_list = [dataset_path + 'SP' + str(i).zfill(2) + '/' for i in test_spk_list]

    test_path_list = []

    if dataset == 'AV_enh':
        for spk_path in test_spk_path_list:
            for noise in noise_test_list:
                for snr in snr_test_list:
                    noise_snrdb = noise + '_' + str(snr).replace('-', 'n') + 'db'
                    # ('', noisy_test_path, '')
                    test_path_list.extend([
                    ('', spk_path + 'audio/noisy_enh/test/' + noise_snrdb + '/', '')
                    ])

    return test_path_list

def prepare_path_list():
    train_path_list = prepare_train_path_list()
    val_path_list = prepare_val_path_list()
    test_path_list = prepare_test_path_list()

    return train_path_list, val_path_list, test_path_list
