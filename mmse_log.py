'''
Based on the implementation by vipchengrui: https://github.com/vipchengrui/traditional-speech-enhancement/blob/master/mmse_log/mmse_log.py
'''

import numpy as np
import wave
import math 
import torchaudio
import scipy.integrate as integ

f = wave.open("../Testset/clean/sp01.wav") # input wave file 

# read format information
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
fs = framerate
# read wave data
str_data = f.readframes(nframes)
# close .wav file
f.close()

# convert waveform data to an array
x = np.fromstring(str_data, dtype=np.short)

print(x)

y_ref, sr = torchaudio.load("../Testset/clean/sp01.wav") #input the clean speech sample
y_ref = y_ref.numpy()[0]
print(y_ref)

# scale to -1.0 -- 1.0
if x.dtype == 'int16':
    nb_bits = 16  # -> 16-bit wav files
elif x.dtype == 'int32':
    nb_bits = 32  # -> 32-bit wav files
max_nb_bit = float(2 ** (nb_bits - 1))
samples = y_ref * (max_nb_bit + 1)


print(samples)

'''
# calculation parameters
len_ = 20 * fs // 1000      # frame size in samples
perc = 50                   # window overlop in percent of frame
len1 = len_ * perc // 100    # overlop'length
len2 = len_ - len1          # window'length - overlop'length

# setting default parameters
aa = 0.98
eta = 0.15
Thres = 3
mu = 0.98
c = np.sqrt(np.pi) / 2
qk = 0.3
qkr = (1 - qk) / qk
ksi_min = 10 ** (-25 / 10)   #-25dB

# hamming window
win = np.hamming(len_)
# normalization gain for overlap+add with 50% overlap
winGain = len2 / sum(win)

# setting inital noise
nFFT = 2 * 2 ** 8
j = 1
noise_mean = np.zeros(nFFT)
for k in range(1, 6):
    noise_mean = noise_mean + abs(np.fft.fft(win * x[j : j + len_] , nFFT))
    j = j + len_
noise_mu = noise_mean / 5
noise_mu2 = noise_mu ** 2

# initialize various variables
k = 1
img = 1j
x_old = np.zeros(len2)
Nframes = len(x) // len2 - 1
xfinal = np.zeros(Nframes * len2)

# === Start Processing ==== #
for n in range(0, Nframes):

    # Windowing
    insign = win * x[k - 1 : k + len_ - 1]

    # Take fourier transform of frame
    spec = np.fft.fft(insign , nFFT)
    sig = abs(spec)
    sig2 = sig ** 2
    # save the noisy phase information
    theta = np.angle(spec)  

    SNRpos = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

    # posteriori SNR
    gammak = np.minimum(sig2 / noise_mu2 , 40) 
    
    # decision-direct estimate of a priori SNR   P231 [7.75]
      
    if n == 0:
        ksi = aa + (1 - aa) * np.maximum(gammak - 1 , 0)
    else:
        ksi = aa * Xk_prev / noise_mu2 + (1 - aa) * np.maximum(gammak - 1 , 0)
        # limit ksi to -25 dB 
        ksi = np.maximum(ksi_min , ksi)  

    # --- implement a simple VAD detector --- #
    if SNRpos < Thres:  # Update noise spectrum
        noise_mu2 = mu * noise_mu2 + (1 - mu) * sig2  # Smoothing processing noise power spectrum
        noise_mu = np.sqrt(noise_mu2)
    
    #Log-MMSE estimator[7.84]
    def integrand(t):
        return np.exp(-t) / t
    A = ksi / (1 + ksi) 
    vk = A * gammak
    ei_vk = np.zeros(len(vk))
    for i in range(len(vk)): 
        ei_vk[i] = 0.5 * integ.quad(integrand,vk[i],np.inf)[0]
    hw = A * np.exp(ei_vk)
 
    # get X(w)
    mmse_speech = hw * sig
    #evk = np.exp(vk)
    #Lambda = qkr * evk / (1 + ksi)
    #pSAP = Lambda / (1 + Lambda)
    #mmse_speech = sig * hw * pSAP

    # save for estimation of a priori SNR in next frame
    Xk_prev = mmse_speech ** 2  

    # IFFT
    x_phase = mmse_speech * np.exp(img * theta)
    xi_w = np.fft.ifft(x_phase , nFFT).real

    # overlop add
    xfinal[k - 1 : k + len2 - 1] = x_old + xi_w[0 : len1]
    x_old = xi_w[len1 + 0 : len_]

    k = k + len2
    
# save wave file
wf = wave.open('out_SNR0_sp01.wav', 'wb')

# setting parameters
wf.setparams(params)
# set waveform file .tostring()Convert array to data
wave_data = (winGain * xfinal).astype(np.short)
wf.writeframes(wave_data.tostring())
# close wave file
wf.close()
'''