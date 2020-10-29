# Pre-require:
#     - Install Octave https://www.gnu.org/software/octave/download.html
#     - Install python package oct2py (@terminal: pip install oct2py)
#     - Install matlab package signal (@octave: pkg install -forge signal)

# Example:
# >>> from NCM_MODULE.Module import NCM

# >>> s = NCM().score('ref_path','deg_path')
# ref_path - Clear input reference speech signal with no noise or distortion
# deg_path - Output signal with noise, distortion, HA gain, and/or processing

from oct2py import octave
import os
import numpy as np

class NCM():
    def __init__(self):
        octave.eval('pkg load signal')
        octave.addpath(os.getcwd()+"/NCM_MODULE")

    def score(self, ref_path, deg_path):
        s = octave.NCM(ref_path, deg_path)
        return s
