import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import wave
import matplotlib
import matplotlib.pyplot as plt

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()
    
def plot_waveform(waveform, sample_rate, path, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    #plt.show()
    plt.savefig(path)

def plot_specgram(waveform, sample_rate, path, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    #plt.show()
    plt.savefig(path)

def plot_spectrogram(spec, path, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    #plt.show()
    plt.savefig(path)

def plot_mel_fbank(path, fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Filter bank')
    axs.imshow(fbank, aspect='auto')
    axs.set_ylabel('frequency bin')
    axs.set_xlabel('mel bin')
    plt.show()
    plt.savefig(path)

def get_spectrogram(path, n_fft = 400, win_len = None, hop_len = None, power = 2.0,):
    waveform, _ = _get_sample(path, resample=resample)
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
        )
    return spectrogram(waveform)

def plot_pitch(waveform, sample_rate, pitch, path):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln2 = axis2.plot(
      time_axis, pitch[0], linewidth=2, label='Pitch', color='green')

  axis2.legend(loc=0)
  plt.show()
  plt.savefig(path)

def plot_kaldi_pitch(waveform, sample_rate, path, pitch, nfcc):
  figure, axis = plt.subplots(1, 1)
  axis.set_title("Kaldi Pitch Feature")
  axis.grid(True)

  end_time = waveform.shape[1] / sample_rate
  time_axis = torch.linspace(0, end_time,  waveform.shape[1])
  axis.plot(time_axis, waveform[0], linewidth=1, color='gray', alpha=0.3)

  time_axis = torch.linspace(0, end_time, pitch.shape[1])
  ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label='Pitch', color='green')
  axis.set_ylim((-1.3, 1.3))

  axis2 = axis.twinx()
  time_axis = torch.linspace(0, end_time, nfcc.shape[1])
  ln2 = axis2.plot(
      time_axis, nfcc[0], linewidth=2, label='NFCC', color='blue', linestyle='--')

  lns = ln1 + ln2
  labels = [l.get_label() for l in lns]
  axis.legend(lns, labels, loc=0)
  plt.show()
  plt.savefig(path)

def normalize_01(x):
    # normalize to -1.0 -- 1.0 
    # Scipy and Wave return integers, which can be normalized and converted to floats according to the number of bits of encoding.
    if x.dtype == 'int16':
        nb_bits = 16  # -> 16-bit wav files
    elif x.dtype == 'int32':
        nb_bits = 32  # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    return (x / (max_nb_bit + 1))
    
def denormalize_01(x):
    # denormalize 
    # SoundFile, Audiolab and Torchaudio returns floats between -1 and 1 (the convention for audio signals), they can be denormalized as well
    if x.dtype == 'int16':
        nb_bits = 16  # -> 16-bit wav files
    elif x.dtype == 'int32':
        nb_bits = 32  # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    return (x * (max_nb_bit + 1))

if __name__ == "__main__":
    #waveform, sample_rate = torchaudio.load("docs/audio/sp15.wav")
    #print_stats(waveform, sample_rate=sample_rate)
    #plot_waveform(waveform, sample_rate, "docs/images/clean_audio_waveform.png")
    #plot_specgram(waveform, sample_rate, "docs/images/clean_audio_spectrogram.png")

    #waveform, sample_rate = torchaudio.load("docs/audio/sp15_station_sn5.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/noisy_audio_waveform.png")
    #plot_specgram(waveform, sample_rate, "docs/images/noisy_audio_spectrogram.png")

    #waveform, sample_rate = torchaudio.load("docs/audio/wiener_filtered_sp15_station_sn5.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/wiener_filtered_audio_waveform.png")
    #plot_specgram(waveform, sample_rate, "docs/images/wiener_filtered_spectrogram.png")

    #waveform, sample_rate = torchaudio.load("docs/audio/ss_filtered_sp15_station_sn5.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/ss_filtered_audio_waveform.png")
    #plot_specgram(waveform, sample_rate, "docs/images/ss_filtered_spectrogram.png")
    
    #waveform, sample_rate = torchaudio.load("docs/audio/MMSE_filtered_sp15_station_sn5.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/mmse_filtered_audio_waveform.png")
    #plot_specgram(waveform, sample_rate, "docs/images/mmse_filtered_spectrogram.png")
    
    #waveform, sample_rate = torchaudio.load("docs/audio/MMSE_Log_filtered_sp15_station_sn5.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/mmse_log_filtered_audio_waveform.png")
    #plot_specgram(waveform, sample_rate, "docs/images/mmse_log_filtered_spectrogram.png")

    #waveform, sample_rate = torchaudio.load("Testset/noise/car_0dB/sp05_car_sn0.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_car_sn0.png")
    #plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_car_sn0.png")

    #waveform, sample_rate = torchaudio.load("Testset/clean/sp05.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/Testset/clean_waveform_sp05.png")
    #plot_specgram(waveform, sample_rate, "docs/images/Testset/clean_spec_sp05.png")

    #waveform, sample_rate = torchaudio.load("Testset/noise/babble_0dB/sp05_babble_sn0.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_babble_sn0.png")
    #plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_babble_sn0.png")

    #waveform, sample_rate = torchaudio.load("Testset/noise/exhibition_0dB/sp05_exhibition_sn0.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_exhibition_sn0.png")
    #plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_exhibition_sn0.png")

    #waveform, sample_rate = torchaudio.load("Testset/noise/restaurant_0dB/sp05_restaurant_sn0.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_restaurant_sn0.png")
    #plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_restaurant_sn0.png")

    #waveform, sample_rate = torchaudio.load("Testset/noise/station_0dB/sp05_station_sn0.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_station_sn0.png")
    #plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_station_sn0.png")

    #waveform, sample_rate = torchaudio.load("Testset/noise/street_0dB/sp05_street_sn0.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_street_sn0.png")
    #plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_street_sn0.png")

    #waveform, sample_rate = torchaudio.load("Testset/noise/train_0dB/sp05_train_sn0.wav")
    #plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_train_sn0.png")
    #plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_train_sn0.png")

    waveform, sample_rate = torchaudio.load("Testset/noise/airport_0dB/sp05_airport_sn0.wav")
    plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_airport_sn0.png")
    plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_airport_sn0.png")

    waveform, sample_rate = torchaudio.load("Testset/noisy/station/sp05_station_sn5.wav")
    plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_station_sn5.png")
    plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_station_sn5.png")

    waveform, sample_rate = torchaudio.load("Testset/noisy/station/sp05_station_sn10.wav")
    plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_station_sn10.png")
    plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_station_sn10.png")

    waveform, sample_rate = torchaudio.load("Testset/noisy/station/sp05_station_sn15.wav")
    plot_waveform(waveform, sample_rate, "docs/images/Testset/waveform_sp05_station_sn15.png")
    plot_specgram(waveform, sample_rate, "docs/images/Testset/spec_sp05_station_sn15.png")