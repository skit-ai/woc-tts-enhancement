# NOIZEUS Dataset

We use the [NOIZEUS](https://ecs.utdallas.edu/loizou/speech/noizeus/) dataset which contains 30 IEEE sentences (produced by three male and three female speakers) corrupted by 8 different real-world noises at different SNRs of 0, 5, 10 and 15 each. The noise includes suburban train noise, babble, car, exhibition hall, restaurant, street, airport and train-station noise. Our testset is created by randomly sampling from the distorted speech samples and keeping the clean speeches intact.

We can visualize some audio samples in our testset by seeing the audio waveforms and the spectrograms and we can categorize our testset into noise event classes as well as SNR ratios.

## Clean audio

<img src="../docs/images/Testset/clean_waveform_sp05.png" width="450"> <img src="../docs/images/Testset/clean_spec_sp05.png" width="450">

## Noise Event 

Let's see the effect of different noise classes on our clean audio(each of them at SNR=0)-

### Babble

<img src="../docs/images/Testset/waveform_sp05_babble_sn0.png" width="450"> <img src="../docs/images/Testset/spec_sp05_babble_sn0.png" width="450">

### Airport

<img src="../docs/images/Testset/waveform_sp05_airport_sn0.png" width="450"> <img src="../docs/images/Testset/spec_sp05_airport_sn0.png" width="450">

### Exhibition

<img src="../docs/images/Testset/waveform_sp05_exhibition_sn0.png" width="450"> <img src="../docs/images/Testset/spec_sp05_exhibition_sn0.png" width="450">

### Street

<img src="../docs/images/Testset/waveform_sp05_street_sn0.png" width="450"> <img src="../docs/images/Testset/spec_sp05_street_sn0.png" width="450">

### Restaurant

<img src="../docs/images/Testset/waveform_sp05_restaurant_sn0.png" width="450"> <img src="../docs/images/Testset/spec_sp05_restaurant_sn0.png" width="450">

### Car

<img src="../docs/images/Testset/waveform_sp05_car_sn0.png" width="450"> <img src="../docs/images/Testset/spec_sp05_car_sn0.png" width="450">

### Train

<img src="../docs/images/Testset/waveform_sp05_train_sn0.png" width="450"> <img src="../docs/images/Testset/spec_sp05_train_sn0.png" width="450">

### Station

<img src="../docs/images/Testset/waveform_sp05_station_sn0.png" width="450"> <img src="../docs/images/Testset/spec_sp05_station_sn0.png" width="450">

## SNR Ratios

Let's us now view the same audio sample recorded within the same noise class (station) and varying SNR ratios-

### SNR = 5 

<img src="../docs/images/Testset/waveform_sp05_station_sn5.png" width="450"> <img src="../docs/images/Testset/spec_sp05_station_sn5.png" width="450">

### SNR = 10

<img src="../docs/images/Testset/waveform_sp05_station_sn10.png" width="450"> <img src="../docs/images/Testset/spec_sp05_station_sn10.png" width="450">

### SNR = 15

<img src="../docs/images/Testset/waveform_sp05_station_sn15.png" width="450"> <img src="../docs/images/Testset/spec_sp05_station_sn15.png" width="450">