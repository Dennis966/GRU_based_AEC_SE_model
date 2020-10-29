# Pre-require:
#     - Install Octave https://www.gnu.org/software/octave/download.html
#     - Install python package oct2py (pip install oct2py)

# Example:
# >>> from SSNR_MODULE.Module import SSNR

# >>> ref, fs = SSNR().read('ref_path') # ref - Clear input reference speech signal with no noise or distortion
# >>> deg, fs = SSNR().read('deg_path') # deg - Output signal with noise, distortion, HA gain, and/or processing
# >>> s = SSNR().score(ref,deg,fs)

from oct2py import octave
import os
import numpy as np

class SSNR():
    def __init__(self):
        octave.addpath(os.getcwd()+"/SSNR_MODULE")

    def read(self, path):
        data, fs = octave.read(path, nout=2)
        return data, fs

    def score(self, clean, noisy, enhan, fs):
        len  = 32 * fs / 1000
        enhan = enhan / (np.std(enhan, axis=0) + 1e-12) * np.std(clean, axis=0) # normalize the energy of the enhanced wav
        s = octave.ssnr(enhan, noisy, clean, len)
        return s
