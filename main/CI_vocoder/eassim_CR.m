function [yout , Srate] = eassim_CR( x, Srate, ha_cutoff, r_nchn, enve_cutoff, flag, vocoder_type, CR)
%function eassim_CR( infile, outfile, ha_cutoff, r_nchn, enve_cutoff, flag, vocoder_type, CR)
% infile:      input file
% outfile:     output file
% ha_cutoff:   lowpass filtering cut off frequency for EAS (HA) 500
% r_nchn:      number of channels used to simulate CI 4
% enve_cutoff: 400
% CR :1 


global center

%[x_in, Srate]= audioread(infile); %wav singal is column wise
%x= x_in;
% % xsize= size( x_in);
% % if xsize( 2)~= 2
% %     error( 'Must be a stereo wav file!\n');
% % end

% !!! think about whether to use this or not, and make sure EAS and BCI use
%           the same configuration
% Preemphasize first
bp = exp(-1200*2*pi/Srate);
ap = exp(-3000*2*pi/Srate);
x = filter([1 -bp], [1 -ap], x);
% !!! 

x_left= x( :, 1);       x_right= x( :, 1);
n_samples= length( x_left);

% lowpass for envelope detection
% lp_cutoff= 500;
lp_cutoff= enve_cutoff;

[bl, al]= butter(2, lp_cutoff/(Srate/2)); 

[r_filterB, r_filterA]= camfilt( Srate, r_nchn, ha_cutoff);

%% LP output for left channel
% 6th order, lowpass for acoustic part with ha_cutoff cutoff frequency

% [bac,aac]= butter(6, ha_cutoff/(Srate/2));
% lp_out= filter( bac, aac, x_left);

x_lp=resample(x_left, ha_cutoff*2, Srate);
x_lp=resample(x_lp, Srate, ha_cutoff*2);
lp_out=x_lp*norm(x_left)/norm(x_lp);
if length(lp_out)>n_samples
    lp_out=lp_out(1:n_samples);
elseif length(lp_out)<n_samples
    lp_out=[lp_out; zeros(n_samples-length(lp_out),1)];
end


%% vocoding output for right channel
r_mpy= randn( n_samples, 1); % for white noise
r_v_out= zeros( n_samples, 1);
% vocoder_type='NV';
% vocoder_type='TV';

for i= 1: r_nchn
    y1= filter( r_filterB(i,:),r_filterA(i,:),x_right);
    ein= norm(y1,2);
    y2= filter(bl, al, abs(y1));
    %% Add CR processing here
    y2=CR*(y2-mean(y2))+mean(y2);
    
    %% noise vocoder
    if strcmp(vocoder_type,'NV')
        y1= y2.* r_mpy; % ----- excite with noise and reconstruct
        y2= filter( r_filterB(i,:), r_filterA(i,:), y1);
    elseif strcmp(vocoder_type,'TV')
    %% tone vocoder    
        tt=[1:1:n_samples]/Srate;
        y2=y2.*sin(2*pi*center(i)*tt');
    end
    
    eout= norm(y2, 2);
    y2= y2* ein/ eout;
    r_v_out= r_v_out+ y2;
end
r_v_out= r_v_out* norm( x_right, 2)/ norm( r_v_out);


%% select output
% yout= [lp_out(:), r_v_out(:)];
if strcmp(flag,'HA+CI')
    yout= lp_out(:)+r_v_out(:);
    yout=yout*norm(x_left)/norm(yout);
elseif strcmp(flag,'HA')
    yout= lp_out(:);
    yout=yout*norm(x_left)/norm(yout);
elseif strcmp(flag,'CI')
    yout= r_v_out(:);
    yout=yout*norm(x_left)/norm(yout);
end

%{
if max( abs( yout))>= 1 % check overflow since random white is involved
    error( 'Overflow detected!\n');
end
%}
%audiowrite(  outfile, yout, Srate);


function [filterB, filterA]= camfilt( Srate, nChannels, ha_cutoff) 
% this is the digital Cambridge filter
global center

FS=Srate/2;
nOrd=6;
UpperFreq=6000;

% LowFreq=80;
LowFreq=ha_cutoff;

upppercam = 21.4*log10(UpperFreq*0.00437+1);
lowcam = 21.4*log10(LowFreq*0.00437+1);
range= upppercam -lowcam;   % cam scale
interval=range/nChannels;
center= zeros(1, nChannels);
upper1= zeros( 1, nChannels);
lower1= zeros( 1, nChannels);

for i=1:nChannels
    upper1(i)=(10^((lowcam+interval*i)/21.4) - 1)/0.00437;
    lower1(i)=(10^((lowcam+interval*(i-1))/21.4) - 1)/0.00437;
    center(i)=0.5*(upper1(i)+lower1(i));
end

if FS<upper1(nChannels), useHigh=1;
else			 useHigh=0;
end

filterA=zeros(nChannels,nOrd+1);
filterB=zeros(nChannels,nOrd+1);

for i=1:nChannels
    W1=[lower1(i)/FS, upper1(i)/FS];
    if i==nChannels
        if useHigh==0
            [b,a]=butter(3,W1);
        else
            [b,a]=butter(6,W1(1),'high');
        end
    else
        [b,a]=butter(3,W1);
    end
    filterB(i,1:nOrd+1)=b;   %----->  Save the coefficients 'b'
    filterA(i,1:nOrd+1)=a;   %----->  Save the coefficients 'a'

    if 0 
        [h,f]=freqz(b,a,512,Srate);
        plot(f,20*log10(abs(h)));
        axis([0 6000 -50 5]);
        hold on;
    end;

end