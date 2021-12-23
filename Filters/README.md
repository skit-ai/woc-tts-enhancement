## Signal Processing Techniques for Speech Enhancement

In this section we explore the various traditional signal processing filters for noise reduction and speech enhancement using the available literature and test them out on samples in our Testset. We visualize our audio samples and the effects our filters have on them by plotting their audio waveforms and spectrograms.

### Clean Audio Sample

<img src="../docs/images/Filters/clean_audio_waveform.png" width="450"> <img src="../docs/images/Filters/clean_audio_spectrogram.png" width="450">

### Noisy Audio Sample (SNR = 15)

<img src="../docs/images/Filters/noisy_audio_waveform.png" width="450"> <img src="../docs/images/Filters/noisy_audio_spectrogram.png" width="450">

### Spectral Subtraction

<img src="../docs/images/Filters/ss_filtered_audio_waveform.png" width="450"> <img src="../docs/images/Filters/ss_filtered_spectrogram.png" width="450"> 

### Bayesian MMSE Filter

<img src="../docs/images/Filters/mmse_filtered_audio_waveform.png" width="450"> <img src="../docs/images/Filters/mmse_filtered_spectrogram.png" width="450"> 

### Bayesian Log MMSE Filter

<img src="../docs/images/Filters/mmse_log_filtered_audio_waveform.png" width="450"> <img src="../docs/images/Filters/mmse_log_filtered_spectrogram.png" width="450">

### Weiner Filter

<img src="../docs/images/Filters/wiener_filtered_audio_waveform.png" width="450"> <img src="../docs/images/Filters/wiener_filtered_spectrogram.png" width="450"> 

### Kalman Filter

<img src="../docs/images/Filters/kalman_filtered_audio_waveform.png" width="450"> <img src="../docs/images/Filters/wiener_filtered_spectrogram.png" width="450"> 