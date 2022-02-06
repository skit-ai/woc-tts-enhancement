import numpy as np
import wave
import scipy.integrate as integ
from scipy.special import iv

def integrand(t):
        return np.exp(-t) / t

class MMSE:

    def __init__(self, aa=0.98, Thres=3, mu=0.98, ksi_min=10 ** (-25 / 10)):
        self.aa = aa
        self.Thres = Thres
        self.mu = mu
        self.ksi_min = ksi_min
    
    def MMSE_Log_path(self, path1, path2):
        # Implemented with the help of https://github.com/vipchengrui/traditional-speech-enhancement/blob/master/mmse_log/mmse_log.py
        # Applies the MMSE Filter on a file present in path1 and a saves the new file in path2
        f = wave.open(path1)
        params = f.getparams()
        nchannels, sampwidth, fs, nframes = params[:4]
        # read wave data
        str_data = f.readframes(nframes)
        # close .wav file
        f.close()
        # convert waveform data to an array
        x = np.fromstring(str_data, dtype=np.short) ## x is denormalized here
        len_ = 20 * fs // 1000      # frame size in samples
        perc = 50                   # window overlop in percent of frame
        len1 = len_ * perc // 100    # overlop'length
        len2 = len_ - len1          # window'length - overlap length
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

        # initialize variables
        k = 1
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
                ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1 , 0)
            else:
                ksi = self.aa * Xk_prev / noise_mu2 + (1 - self.aa) * np.maximum(gammak - 1 , 0)
                ksi = np.maximum(self.ksi_min , ksi)  

            # --- implement a simple VAD detector --- #
            if SNRpos < self.Thres:  # Update noise spectrum
                noise_mu2 = self.mu * noise_mu2 + (1 - self.mu) * sig2  # Smoothing processing noise power spectrum
                noise_mu = np.sqrt(noise_mu2)
    
            #Log-MMSE estimator[7.84]
            A = ksi / (1 + ksi) 
            vk = A * gammak
            ei_vk = np.zeros(len(vk))
            for i in range(len(vk)): 
                ei_vk[i] = 0.5 * integ.quad(integrand,vk[i],np.inf)[0]
            hw = A * np.exp(ei_vk)
 
            # get X(w)
            mmse_speech = hw * sig

            # save for estimation of a priori SNR in next frame
            Xk_prev = mmse_speech ** 2  

            # IFFT
            x_phase = mmse_speech * np.exp(1j * theta)
            xi_w = np.fft.ifft(x_phase , nFFT).real

            # overlop add
            xfinal[k - 1 : k + len2 - 1] = x_old + xi_w[0 : len1]
            x_old = xi_w[len1 + 0 : len_]

            k = k + len2
        wf = wave.open(path2, 'wb')
        # setting parameters
        wf.setparams(params)
        # set waveform file .tostring()Convert array to data
        wave_data = (winGain * xfinal).astype(np.short)
        wf.writeframes(wave_data.tostring())
        # close wave file
        wf.close()
    
    def MMSE_path(self, path1, path2):
        # Implemented with the help of https://github.com/vipchengrui/traditional-speech-enhancement/blob/master/mmse/mmse.py
        # Applies the MMSE Filter on a file present in path1 and a saves the new file in path2
        f = wave.open(path1)
        params = f.getparams()
        nchannels, sampwidth, fs, nframes = params[:4]
        # read wave data
        str_data = f.readframes(nframes)
        # close .wav file
        f.close()
        # convert waveform data to an array
        x = np.fromstring(str_data, dtype=np.short) ## x is denormalized here
        len_ = 20 * fs // 1000      # frame size in samples
        perc = 50                   # window overlop in percent of frame
        len1 = len_ * perc // 100    # overlop'length
        len2 = len_ - len1          # window'length - overlap length
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

        # initialize variables
        k = 1
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
                ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1 , 0)
            else:
                ksi = self.aa * Xk_prev / noise_mu2 + (1 - self.aa) * np.maximum(gammak - 1 , 0)
                ksi = np.maximum(self.ksi_min , ksi)  

            # --- implement a simple VAD detector --- #
            if SNRpos < self.Thres:  # Update noise spectrum
                noise_mu2 = self.mu * noise_mu2 + (1 - self.mu) * sig2  # Smoothing processing noise power spectrum
                noise_mu = np.sqrt(noise_mu2)
    
            #Log-MMSE estimator[7.84]
            # [7.40]
            c = c = np.sqrt(np.pi) / 2
            vk = gammak * ksi / (1 + ksi)
            # the modified Bessel function of n order iv(n,x)
            j_0 = iv(0 , vk/2) #modified bessel function of the first kind of real order 
            j_1 = iv(1 , vk/2)    
            C = np.exp(-0.5 * vk)
            A = ((c * (vk ** 0.5)) * C) / gammak      #[7.40] A
            B = (1 + vk) * j_0 + vk * j_1             #[7.40] B
            hw = A * B                                #[7.40]

            # get X(w)
            mmse_speech = hw * sig

            # save for estimation of a priori SNR in next frame
            Xk_prev = mmse_speech ** 2  

            # IFFT
            x_phase = mmse_speech * np.exp(1j * theta)
            xi_w = np.fft.ifft(x_phase , nFFT).real

            # overlop add
            xfinal[k - 1 : k + len2 - 1] = x_old + xi_w[0 : len1]
            x_old = xi_w[len1 + 0 : len_]

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
    path1 = "../docs/audio/station/sp15_station_sn5.wav"
    path2 = "../docs/audio/MMSE_Log_filtered_sp15_station_sn5.wav"
    path3 = "../docs/audio/MMSE_filtered_sp15_station_sn5.wav"
    mmse_filter = MMSE()
    mmse_filter.MMSE_Log_path(path1, path2)
    mmse_filter.MMSE_path(path1, path3)
    print("Audio denoised!")
