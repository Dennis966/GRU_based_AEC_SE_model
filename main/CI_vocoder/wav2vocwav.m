function wav2vocwav(x_path, y_path)

ha_cutoff = 500;
r_nchn = 4;
enve_cutoff = 400;
flag = 'CI';
vocoder_type = 'TV';
CR = 1;

[x, Srate] = audioread(x_path);
[y, Srate] = eassim_CR(x, Srate, ha_cutoff, r_nchn, enve_cutoff, flag, vocoder_type, CR);

audiowrite(y_path, y, Srate);