import numpy as np
import wave

def get_mel(SNR, mel_max=10):
    s = 25 / (mel_max - 1)
    mel_0 = (1 + 4 * mel_max) / 5
    if -5.0 <= SNR <= 20.0:
        a = mel_0 - SNR / s
    else:
        if SNR < -5.0:
            a = mel_max
        if SNR > 20:
            a = 1
    return a

class Wiener:
    
    def __init__(self, Thres=3, Expnt=1.0, G=0.9, mel_max=10):
        self.Thres = Thres
        self.Expnt = Expnt
        self.G = G
        self.mel_max = mel_max
    
    def Wiener_path(self, path1, path2):
        # # Implemented with the help of: https://github.com/vipchengrui/traditional-speech-enhancement/blob/master/wiener_filtering/wiener_filtering_phase.py
        # input wave file 
        f = wave.open(path1)

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

        # noisy speech FFT
        x_FFT = abs(np.fft.fft(x))

        # calculation parameters
        len_ = 20 * fs // 1000      # frame size in samples
        perc = 50                   # window overlop in percent of frame
        len1 = len_ * perc // 100   # overlop'length
        len2 = len_ - len1          # window'length - overlop'length

        # hamming window
        #win = np.hamming(len_)

        # sine window
        i = np.linspace(0,len_ - 1,len_)
        win = np.sqrt(2/(len_ + 1)) * np.sin(np.pi * (i + 1) / (len_ + 1))

        # normalization gain for overlap+add with 50% overlap
        winGain = len2 / sum(win)

        # nFFT = 2 * 2 ** (nextpow2.nextpow2(len_))
        nFFT = 2 * 2 ** 8
        noise_mean = np.zeros(nFFT)
        j = 1
        for k in range(1, 6):
            noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
            j = j + len_
        noise_mu = noise_mean / 5

        # initialize various variables
        k = 1
        x_old = np.zeros(len1)
        Nframes = len(x) // len2 - 1
        xfinal = np.zeros(Nframes * len2)

        # === Start Processing ==== #
        for n in range(0, Nframes):

            # Windowing
            insign = win * x[k-1:k + len_ - 1]    
            # compute fourier transform of a frame
            spec = np.fft.fft(insign, nFFT)    
            # compute the magnitude
            sig = abs(spec)     
            # save the noisy phase information
            theta = np.angle(spec)  
            # Posterior SNR
            SNRpos = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)
    
            # --- wiener filtering --- #

            # 1 spectral subtraction(Half wave rectification)
            sub_speech = sig ** self.Expnt - noise_mu ** self.Expnt
            #      When the pure signal is less than the noise signal power
            diffw = sig ** self.Expnt - noise_mu ** self.Expnt   
            #   beta negative components
            def find_index(x_list):
                index_list = []
                for i in range(len(x_list)):
                    if x_list[i] < 0:
                        index_list.append(i)
                return index_list
            z = find_index(diffw)
            if len(z) > 0:
                sub_speech[z] = 0
    
            # Priori SNR
            SNRpri = 10 * np.log10(np.linalg.norm(sub_speech, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

            mel = get_mel(SNRpri, self.mel_max) 

            # 2 gain function Gk
            #G_k = (sig ** Expnt - noise_mu ** Expnt) / sig ** Expnt
            G_k = sub_speech ** 2 / (sub_speech ** 2 + mel * noise_mu ** 2)
            #wf_speech = G_k * sub_speech ** (1 / Expnt)
            wf_speech = G_k * sig
    
            # --- implement a simple VAD detector --- #
            if SNRpos < self.Thres:  # Update noise spectrum
                noise_temp = self.G * noise_mu ** self.Expnt + (1 - self.G) * sig ** self.Expnt  # Smoothing processing noise power spectrum
                noise_mu = noise_temp ** (1 / self.Expnt)  # New noise amplitude spectrum
    
            # add phase    
            #wf_speech[nFFT // 2 + 1:nFFT] = np.flipud(wf_speech[1:nFFT // 2])
            x_phase = wf_speech * np.exp(1j * theta)
    
            # take the IFFT
            xi = np.fft.ifft(x_phase).real
    
            # --- Overlap and add --- #
            xfinal[k-1:k + len2 - 1] = x_old + xi[0:len1]
            x_old = xi[0 + len1:len_]

            k = k + len2

        # save wave file
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
    path2 = "../docs/audio/wiener_filtered_sp15_station_sn5.wav"
    wiener_filter = Wiener()
    wiener_filter.Wiener_path(path1, path2)
