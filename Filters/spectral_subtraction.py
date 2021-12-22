import numpy as np
import wave
from noise_estimation import Init_MCRA2
from noise_estimation import Est_MCRA2

# setting SNR
def berouti(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 4 - SNR * 3 / 20
    else:
        if SNR < -5.0:
            a = 5
        if SNR > 20:
            a = 1
    return a

def berouti1(SNR):
    if -5.0 <= SNR <= 20.0:
        a = 3 - SNR * 2 / 20
    else:
        if SNR < -5.0:
            a = 4
        if SNR > 20:
            a = 1
    return a

class Spectral_Subtraction:

  def __init__ (self, Thres=3, Expnt=2.0, beta=0.002):
      self.Thres = Thres   # VAD threshold in dB SNRseg
      self.Expnt = Expnt   # exp(Expnt)
      #self.G     = G
      self.beta  = beta
  
  def Spectral_Subtraction_path(self, path1, path2):
      # spectral subtraction along with noise estimation
      # Implemented with the help of https://github.com/vipchengrui/traditional-speech-enhancement/blob/master/spectral_subtraction/spectral_subtraction.py
      f = wave.open(path1)
      # read format information
      # # (nchannels, sampwidth, framerate, nframes, comptype, compname)
      params = f.getparams()
      nchannels, sampwidth, framerate, nframes = params[:4]
      fs = framerate
      # read wave data
      str_data = f.readframes(nframes)
      # close .wav file
      f.close()
      # convert waveform data to an array
      x = np.fromstring(str_data, dtype=np.short)
      # noisy speech FFT
      x_FFT = abs(np.fft.fft(x))
      # calculation parameters
      len_ = 20 * fs // 1000      # frame size in samples
      perc = 50                   # window overlop in percent of frame
      len1 = len_ * perc // 100   # overlop'length
      len2 = len_ - len1          # window'length - overlop'length
      # initial Hamming window
      win = np.hamming(len_)
      # normalization gain for overlap+add with 50% overlap
      winGain = len2 / sum(win)
      # nFFT = 2 * 2 ** (nextpow2.nextpow2(len_))
      nFFT = 2 * 2 ** 8
      # initialize various variables
      k = 1
      img = 1j
      x_old = np.zeros(len1)
      Nframes = len(x) // len2 - 1
      xfinal = np.zeros(Nframes * len2)

      for n in range(0, Nframes):
          # Windowing
            insign = win * x[k-1:k + len_ - 1]   
          # compute fourier transform of a frame
            spec = np.fft.fft(insign, nFFT)   
          # compute the magnitude
            sig = abs(spec)
          # noisy speech power spec
            ns_ps = sig ** 2
          # save the noisy phase information
            theta = np.angle(spec)
          # Noise Estimation
          # #Init_Weight、ConMinTrack、MCRA、MCRA2
            if n == 0:
                para = Init_MCRA2(ns_ps,fs).info()    
            else:
                para = Est_MCRA2(ns_ps,para).est()

            noise_ps = para['noise_ps']
            noise_mu = np.sqrt(noise_ps)

            # SNR
            SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)
            # setting alpha
            if self.Expnt == 1.0:     # magnitude spectrum
                alpha = berouti1(SNRseg)
            else:                # power spectrum
                alpha = berouti(SNRseg)
    
            # --- over subtraction --- #
            sub_speech = sig ** self.Expnt - alpha * noise_mu ** self.Expnt;
            # the pure signal is less than the noise signal power
            diffw = sub_speech - self.beta * noise_mu ** self.Expnt
            # beta negative components
            def find_index(x_list):
                index_list = []
                for i in range(len(x_list)):
                    if x_list[i] < 0:
                        index_list.append(i)
                return index_list
            z = find_index(diffw)
            if len(z) > 0:
                # The lower bound is represented by the estimated noise signal
                sub_speech[z] = self.beta * noise_mu[z] ** self.Expnt
    
            # add phase
            #sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
            #x_phase = (sub_speech ** (1 / Expnt)) * (np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))
            x_phase = (sub_speech ** (1 / self.Expnt)) * np.exp(img * theta)
    
            # take the IFFT
            xi = np.fft.ifft(x_phase).real

            # --- Overlap and add --- #
            xfinal[k-1:k + len2 - 1] = x_old + xi[0:len1]
            x_old = xi[0 + len1:len_]

            k = k + len2
            wf = wave.open(path2, 'wb')
            # setting parameters
            wf.setparams(params)
            # set waveform file .tostring()Convert array to data
            wave_data = (winGain * xfinal).astype(np.short)
            wf.writeframes(wave_data.tostring())
            # close wave file
            wf.close()

if __name__ == "__main__":
    path1 = "../Testset/noisy/station/sp15_station_sn5.wav"
    path2 = "../docs/audio/ss_filtered_sp15_station_sn5.wav"
    ss_filter = Spectral_Subtraction()
    ss_filter.Spectral_Subtraction_path(path1, path2)