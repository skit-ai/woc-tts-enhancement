import numpy as np
import scipy.signal as signal
import librosa

def resample(input_signal, old_sample_rate, new_sample_rate):
  """
  The input_signal is upsampled by the factor up, a zero-phase low-pass FIR filter is applied, and then it 
  is downsampled by the factor down. The resulting sample rate is up / down times the original sample rate. 
  By default, values beyond the boundary of the signal are assumed to be zero during the filtering step.
  """
  resampled_signal = signal.resample_poly(input_signal, new_sample_rate, old_sample_rate)
  return resampled_signal.astype(input_signal.dtype)

def stft(audio, dimension):
  """
  The STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) 
  over short overlapping windows of size=2048. This function deals with both cases of when our speech is mono as
  well when it is sterio (multiple channels).
  """
  dims = audio.ndim
  transform = np.array([], ndim=1)
  if(dims==1):
    transform = librosa.stft(audio) #mono case
  else:
    transform = librosa.stft(audio[:,dimension]) #sterio case
  return transform

def spectral_subtraction(noise_profile_n, input_signal_y, dimension):
  """
  This function removes a noise profile from a given input signal. This is done by first applying STFT on both the 
  noise as well as the input signal, subtracting the noise magnitude from the input, combining the phase information 
  and applying the inverse STFT.
  """
  N = stft(noise_profile_n, dimension)
  noise_mag = np.abs(N) #magnitude spectrum

  Y = stft(input_signal_y, dimension)
  input_mag = np.abs(Y) 

  phase_spectrum = np.angle(Y) #phase spectrum
  phase_info = np.exp((1.0j)*phase_spectrum) #phase information

  noise_mean = np.mean(noise_mag, axis=1, dtype="float64")
  noise_mean = noise_mean[:, np.newxis]

  output_X = input_mag - noise_mean
  X = np.clip(output_X, a_min=0.0, a_max=None)

  X = X * phase_info
  output_x = librosa.istft(X)

  return output_x
