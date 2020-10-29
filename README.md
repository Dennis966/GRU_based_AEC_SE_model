# Modified SE-AEC Model (GRU)

## A GRU based Speech Enhancement Autoencoder Model.

@Since I had problems importing SRU's, I've replaced the SRU units with GRU units. I commented out all the SRU-based models in `build.py` and declared my new models in `main_SE.py`, `main_AEC.py` and `main_SE_fixAEC.py`. A discription of the new models is shown below.

### SE model

* Model Name: First_SE_model_4
* *Structure:* 4 BGRU layers (Bidirectional GRU) 
* Question: In the original code, we have layer normalization for SRU units. However, we don't have layer normalization for GRU's. Is this going to be a problem? Same with AEC model.

### AEC model

* Model Name: First_AEC_model_4
* Structure: 4 BGRU layers (Bidirectional GRU)
* Question: In the original code, I see that some classes are defined as 4 layers of SRU's, some are defined as layers of CNN's concatenated with SRU's. For simplicity, I only defined my AEC model to be 4 layers of GRU's. Is it better to use a combination CNN's and recurrent neural networks? 

### SE_fixAEC model

* Model Name: First_SE_fix_AEC_model_4
* Structure: 8 GRU layers. (A concatenation of the previous SE model and AEC model)


## Bash files
### 1.`my_first_SE_bash.sh`
Trains and tests the SE model.
(Seems to work, no compiling errors, but the results don't seem to be good)
### 2.`my_first_bash.sh`
Trains and tests the AEC model first, then trains and tests the SE_fixAEC model.
(AEC model seems fine, but problem occurs when we start training SE_fixAEC model)

