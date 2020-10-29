import torch
import torch.nn as nn
## 9/13 commented out         from sru import SRU

layer_group = (nn.Conv1d, nn.Linear)

# Weight Initialize
def weights_init(m):
    # if isinstance(m, layer_group):
    #     nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu', 0.3))

    if isinstance(m, layer_group):
        # print('m =', m)
        for name, param in m.named_parameters():
            if 'weight' in name:
                # print('name =', name)
                nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('leaky_relu', 0.3))

class Conv(nn.Module):

    def __init__(self, in_chan, out_chan, kernal, padding):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv1d(in_chan, out_chan, kernal, padding=padding),
                    nn.InstanceNorm1d(num_features=out_chan),
                    nn.LeakyReLU(negative_slope=0.3)
                    )

    def forward(self, x):
        # x: (bs, 257, len)
        x = self.conv(x)
        return x

# class Conv_tanh(nn.Module):

#     def __init__(self, in_chan, out_chan, kernal, padding):
#         super(Conv_tanh, self).__init__()
#         self.conv = nn.Sequential(
#                     nn.Conv1d(in_chan, out_chan, kernal, padding=padding),
#                     nn.InstanceNorm1d(num_features=out_chan),
#                     nn.Tanh()
#                     )

#     def forward(self, x):
#         # x: (bs, 257, len)
#         x = self.conv(x)
#         return x

'''
class Bsru(nn.Module):

    def __init__(self, in_chan, out_chan, kernal, num_layers):
        super(Bsru, self).__init__()

        self.sru = SRU(in_chan, kernal, num_layers=num_layers, dropout=0, layer_norm=True, bidirectional=True, rescale=True)
        self.hidden2out = nn.Sequential(
            nn.Linear(2*kernal, out_chan),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (bs, 257, len)
        x = x.permute(2, 0, 1) # (len, bs, 257)
        x, _ = self.sru(x)
        x = self.hidden2out(x)
        x = x.permute(1, 2, 0) # (bs, 257, len)
        return x
'''

######################################################### My Basic GRU Units ##########################################################
#######################################################################################################################################

class BGRU(nn.Module):

    def __init__(self, in_chan, out_chan, kernal, num_layers):                              # kernal = hidden layer size
        super(BGRU, self).__init__()
                                                                                            # no normalization ??
        self.gru = nn.GRU(input_size = in_chan, hidden_size = kernal, num_layers = num_layers, dropout = 0, bidirectional = True)
        self.hidden2out = nn.Sequential(
            nn.Linear(2*kernal, out_chan),                 # Since we use a "bidirectional" GRU, our output size of self.gru = 2*kernal
            nn.ReLU()
        )


    def forward(self, x):
        x = x.permute(2, 0, 1)                          # x: (bs, 257, len).  After permute(), x:(len, bs, 257). Why is this order correct?
        x, _ = self.gru(x)
        x = self.hidden2out(x)
        x = x.permute(1, 2, 0)                          
        return x                                        # Now, x:(bs, 257, len), which has the same dim order as that of the input. Why?


#######################################################################################################################################
#######################################################################################################################################

# class Bsru_leaky(nn.Module):

#     def __init__(self, in_chan, out_chan, kernal, num_layers):
#         super(Bsru_leaky, self).__init__()

#         self.sru = SRU(in_chan, kernal, num_layers=num_layers, dropout=0, layer_norm=True, bidirectional=True)
#         self.hidden2out = nn.Sequential(
#             nn.Linear(2*kernal, out_chan),
#             nn.LeakyReLU()
#         )

#     def forward(self, x):
#         # x: (bs, 257, len)
#         x = x.permute(2, 0, 1) # (len, bs, 257)
#         x, _ = self.sru(x)
#         x = self.hidden2out(x)
#         x = x.permute(1, 2, 0) # (bs, 257, len)
#         return x

# class Bsru_softplus(nn.Module):

#     def __init__(self, in_chan, out_chan, kernal, num_layers):
#         super(Bsru_softplus, self).__init__()

#         self.sru = SRU(in_chan, kernal, num_layers=num_layers, dropout=0, layer_norm=True, bidirectional=True)
#         self.hidden2out = nn.Sequential(
#             nn.Linear(2*kernal, out_chan),
#             nn.Softplus()
#         )

#     def forward(self, x):
#         # x: (bs, 257, len)
#         x = x.permute(2, 0, 1) # (len, bs, 257)
#         x, _ = self.sru(x)
#         x = self.hidden2out(x)
#         x = x.permute(1, 2, 0) # (bs, 257, len)
#         return x

################################################## SE Model ##################################################

'''
class SE_BSRU(nn.Module):

    def __init__(self):
        super(SE_BSRU, self).__init__()
        self.bsru = Bsru(257, 257, 512, num_layers=6)

    def forward(self, x):
        x = self.bsru(x)
        return x

class SE_BSRU4(nn.Module):

    def __init__(self):
        super(SE_BSRU4, self).__init__()
        self.bsru = Bsru(257, 257, 512, num_layers=4)

    def forward(self, x):
        x = self.bsru(x)
        return x

class SE_BSRU8(nn.Module):

    def __init__(self):
        super(SE_BSRU8, self).__init__()
        self.bsru = Bsru(257, 257, 512, num_layers=8)

    def forward(self, x):
        x = self.bsru(x)
        return x
'''

################################################## My SE model ################################################
###############################################################################################################

class First_SE_model_4(nn.Module):                             # 4 GRU LAYERS

    def __init__(self):
        super(First_SE_model_4, self).__init__()
        self.bgru = BGRU(257, 257, 512, num_layers = 4)

    def forward(self, x):
        x = self.bgru(x)
        return x

###############################################################################################################
###############################################################################################################

################################################## AEC Model ##################################################

# AEC_DENS activated 9/13

class AEC_DENS(nn.Module):
    def __init__(self):
        super(AEC_DENS, self).__init__()
        in_chan  = 257
        out_chan = 257
        kernal   = 512

        self.dens = nn.Sequential(
                    nn.Linear(in_chan,   kernal),
                    nn.ReLU(),
                    nn.Linear( kernal,   kernal),
                    nn.ReLU(),
                    nn.Linear( kernal, out_chan),
                    nn.ReLU()
                    )

    def forward(self, x):
     #   x: (bs, 257, len)
        x = x.permute(2, 0, 1) # (len, bs, 257)
        x = self.dens(x)
        x = x.permute(1, 2, 0) # (bs, 257, len)
        return x

# class AEC_DENS4(nn.Module):

#     def __init__(self):
#         super(AEC_DENS4, self).__init__()
#         in_chan  = 257
#         out_chan = 257
#         kernal   = 512

#         self.dens = nn.Sequential(
#                     nn.Linear(in_chan,   kernal),
#                     nn.ReLU(),
#                     nn.Linear( kernal,   kernal),
#                     nn.ReLU(),
#                     nn.Linear( kernal,   kernal),
#                     nn.ReLU(),
#                     nn.Linear( kernal, out_chan),
#                     nn.ReLU()
#                     )

#     def forward(self, x):
#         # x: (bs, 257, len)
#         x = x.permute(2, 0, 1) # (len, bs, 257)
#         x = self.dens(x)
#         x = x.permute(1, 2, 0) # (bs, 257, len)
#         return x
'''
class AEC_BSRU(nn.Module):

    def __init__(self):
        super(AEC_BSRU, self).__init__()
        self.bsru = Bsru(257, 257, 512, num_layers=6)

    def forward(self, x):
        x = self.bsru(x)
        return x
'''
# class AEC_BSRU_256(nn.Module):

#     def __init__(self):
#         super(AEC_BSRU_256, self).__init__()
#         self.bsru = Bsru(257, 257, 256, num_layers=6)

#     def forward(self, x):
#         x = self.bsru(x)
#         return x

# class AEC_BSRU_400(nn.Module):

#     def __init__(self):
#         super(AEC_BSRU_400, self).__init__()
#         self.bsru = Bsru(257, 257, 400, num_layers=6)

#     def forward(self, x):
#         x = self.bsru(x)
#         return x

# class AEC_BSRU_500(nn.Module):

#     def __init__(self):
#         super(AEC_BSRU_500, self).__init__()
#         self.bsru = Bsru(257, 257, 500, num_layers=6)

#     def forward(self, x):
#         x = self.bsru(x)
#         return x

# class AEC_BSRU_520(nn.Module):

#     def __init__(self):
#         super(AEC_BSRU_520, self).__init__()
#         self.bsru = Bsru(257, 257, 520, num_layers=6)

#     def forward(self, x):
#         x = self.bsru(x)
#         return x

# class AEC_BSRU_550(nn.Module):

#     def __init__(self):
#         super(AEC_BSRU_550, self).__init__()
#         self.bsru = Bsru(257, 257, 550, num_layers=6)

#     def forward(self, x):
#         x = self.bsru(x)
#         return x

# class AEC_BSRU_600(nn.Module):

#     def __init__(self):
#         super(AEC_BSRU_600, self).__init__()
#         self.bsru = Bsru(257, 257, 600, num_layers=6)

#     def forward(self, x):
#         x = self.bsru(x)
#         return x

# class AEC_BSRU_1024(nn.Module):

#     def __init__(self):
#         super(AEC_BSRU_1024, self).__init__()
#         self.bsru = Bsru(257, 257, 1024, num_layers=6)

#     def forward(self, x):
#         x = self.bsru(x)
#         return x

# class AEC_BSRU_LEAKY(nn.Module):

#     def __init__(self):
#         super(AEC_BSRU_LEAKY, self).__init__()
#         self.bsru = Bsru_leaky(257, 257, 512, num_layers=6)

#     def forward(self, x):
#         x = self.bsru(x)
#         return x

# class AEC_BSRU_SOFTPLUS(nn.Module):

#     def __init__(self):
#         super(AEC_BSRU_SOFTPLUS, self).__init__()
#         self.bsru = Bsru_softplus(257, 257, 512, num_layers=6)

#     def forward(self, x):
#         x = self.bsru(x)
#         return x
'''
class AEC_BSRU4(nn.Module):

    def __init__(self):
        super(AEC_BSRU4, self).__init__()
        self.bsru = Bsru(257, 257, 512, num_layers=4)

    def forward(self, x):
        x = self.bsru(x)
        return x

class AEC_BSRU8(nn.Module):

    def __init__(self):
        super(AEC_BSRU8, self).__init__()
        self.bsru = Bsru(257, 257, 512, num_layers=4)

    def forward(self, x):
        x = self.bsru(x)
        return x

class AEC_BSRU20(nn.Module):

    def __init__(self):
        super(AEC_BSRU20, self).__init__()
        self.bsru = Bsru(257, 257, 512, num_layers=4)

    def forward(self, x):
        x = self.bsru(x)
        return x
'''
# class AEC_BSRUDENS(nn.Module):

#     def __init__(self):
#         super(AEC_BSRUDENS, self).__init__()
#         in_chan  = 257
#         out_chan = 257
#         kernal   = 512

#         self.bsru = SRU(in_chan, kernal, num_layers=6, dropout=0, layer_norm=True, bidirectional=True)
#         self.dens = nn.Sequential(
#                     nn.Linear(2*kernal, kernal),
#                     nn.ReLU(),
#                     nn.Linear(kernal, kernal),
#                     nn.ReLU(),
#                     nn.Linear(kernal, out_chan),
#                     nn.ReLU()
#                     )

#     def forward(self, x):
#         x = x.permute(2, 0, 1) # (len, bs, 257)
#         x, _ = self.bsru(x)
#         x = self.dens(x)
#         x = x.permute(1, 2, 0) # (bs, 257, len)
#         return x
'''
class AEC_CSRU(nn.Module):

    def __init__(self):
        super(AEC_CSRU, self).__init__()
        self.conv = nn.Sequential(
            Conv(257, 257, 15, padding=7),
            Conv(257, 257, 15, padding=7)
        )
        self.bsru = Bsru(257, 257, 512, num_layers=6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bsru(x)
        return x

class AEC_CSRU3(nn.Module):

    def __init__(self):
        super(AEC_CSRU3, self).__init__()
        self.conv = nn.Sequential(
            Conv(257, 257, 15, padding=7),
            Conv(257, 257, 15, padding=7),
            Conv(257, 257, 15, padding=7)
        )
        self.bsru = Bsru(257, 257, 512, num_layers=6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bsru(x)
        return x
'''
# class AEC_CSRU3_TANH(nn.Module):

#     def __init__(self):
#         super(AEC_CSRU3_TANH, self).__init__()
#         self.conv = nn.Sequential(
#             Conv_tanh(257, 257, 15, padding=7),
#             Conv_tanh(257, 257, 15, padding=7),
#             Conv_tanh(257, 257, 15, padding=7)
#         )
#         self.bsru = Bsru(257, 257, 512, num_layers=6)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bsru(x)
#         return x
'''
class AEC_CSRU4(nn.Module):

    def __init__(self):
        super(AEC_CSRU4, self).__init__()
        self.conv = nn.Sequential(
            Conv(257, 257, 15, padding=7),
            Conv(257, 257, 15, padding=7),
            Conv(257, 257, 15, padding=7),
            Conv(257, 257, 15, padding=7)
        )
        self.bsru = Bsru(257, 257, 512, num_layers=6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bsru(x)
        return x

class AEC_CSRU3_3(nn.Module):

    def __init__(self):
        super(AEC_CSRU3_3, self).__init__()
        k = 3
        p = k // 2

        self.conv = nn.Sequential(
            Conv(257, 257, k, padding=p),
            Conv(257, 257, k, padding=p),
            Conv(257, 257, k, padding=p)
        )
        self.bsru = Bsru(257, 257, 512, num_layers=6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bsru(x)
        return x

class AEC_CSRU3_5(nn.Module):

    def __init__(self):
        super(AEC_CSRU3_5, self).__init__()
        k = 5
        p = k // 2

        self.conv = nn.Sequential(
            Conv(257, 257, k, padding=p),
            Conv(257, 257, k, padding=p),
            Conv(257, 257, k, padding=p)
        )
        self.bsru = Bsru(257, 257, 512, num_layers=6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bsru(x)
        return x

class AEC_CSRU3_7(nn.Module):

    def __init__(self):
        super(AEC_CSRU3_7, self).__init__()
        k = 7
        p = k // 2

        self.conv = nn.Sequential(
            Conv(257, 257, k, padding=p),
            Conv(257, 257, k, padding=p),
            Conv(257, 257, k, padding=p)
        )
        self.bsru = Bsru(257, 257, 512, num_layers=6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bsru(x)
        return x

class AEC_CSRU3_9(nn.Module):

    def __init__(self):
        super(AEC_CSRU3_9, self).__init__()
        k = 9
        p = k // 2

        self.conv = nn.Sequential(
            Conv(257, 257, k, padding=p),
            Conv(257, 257, k, padding=p),
            Conv(257, 257, k, padding=p)
        )
        self.bsru = Bsru(257, 257, 512, num_layers=6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bsru(x)
        return x
'''
################################################# My AEC Model ###################################################
##################################################################################################################

class First_AEC_model_4_problem(nn.Module):

    def __init__(self):
        super(First_AEC_model_4_problem, self).__init__()
        k = 4                                                    ### k = Size of convolving kernal
        p = k // 2                                               ### Why do we pad half of k? What benefits do we get from padding? Faster?

        self.conv = nn.Sequential(
            Conv(257, 257, k, padding = p),
            Conv(257, 257, k, padding = p),
            Conv(257, 257, k, padding = p),
            Conv(257, 257, k, padding = p)
        )
        self.bgru = BGRU(257, 257, 512, num_layers = 6)          
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bgru(x)

        return x 

class First_AEC_model_4(nn.Module):

    def __init__(self):
        super(First_AEC_model_4, self).__init__()
        self.bgru = BGRU(257, 257, 512, num_layers = 4)

    def forward(self, x):
        x = self.bgru(x)

        return x

##################################################################################################################
##################################################################################################################

################################################## SE-AEC Model ##################################################

'''
class BSRU(nn.Module):

    def __init__(self):
        super(BSRU, self).__init__()
        self.se = SE_BSRU()
        self.aec = AEC_BSRU()

    def forward(self, wav):
        enhan = self.se(wav)
        ci = self.aec(enhan)
        return enhan, ci

class BSRU4(nn.Module):

    def __init__(self):
        super(BSRU4, self).__init__()
        self.se = SE_BSRU4()
        self.aec = AEC_BSRU()

    def forward(self, wav):
        enhan = self.se(wav)
        ci = self.aec(enhan)
        return enhan, ci

class BSRU8(nn.Module):

    def __init__(self):
        super(BSRU8, self).__init__()
        self.se = SE_BSRU8()
        self.aec = AEC_BSRU()

    def forward(self, wav):
        enhan = self.se(wav)
        ci = self.aec(enhan)
        return enhan, ci
'''

################################################# My SE-AEC Model ################################################
##################################################################################################################

class First_SE_fix_AEC_model_4(nn.Module):

    def __init__(self):
        super(First_SE_fix_AEC_model_4, self).__init__()
        self.se = First_SE_model_4()
        self.aec = First_AEC_model_4()

        for param in self.aec.parameters():
            param.requires_grad = False

    def forward(self, wav):
        enhan = self.se(wav)
        ci = self.aec(enhan)
        
        return enhan, ci

##################################################################################################################



