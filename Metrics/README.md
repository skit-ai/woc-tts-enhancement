## Metrics

These are some of the metrics implemented in this repository as well as tested out from various sources. Their usage has been demonstrated in the examples.py file. This is how these metrics perform on some samples-


```python
>>>import torchaudio
>>>import numpy as np
>>>path1 = "../Testset/clean/sp01.wav" #input the clean audio
>>>path2 = "../Testset/noisy/babble/sp01_babble_sn10.wav" #input the corresponding noisy audio
>>>y_ref, sr_ref = torchaudio.load(path1) #load the files
>>>y_syn, sr_syn = torchaudio.load(path2)
```

### F0 Frame Error Rate (FER)

```python
>>>from Metrics.f0_frame_error import FFE
>>>ffe = FFE(22050)
>>>print((ffe.calculate_ffe(y_ref, y_syn))
0.11494252873563218
```

### MSD

```python
>>>from Metrics.msd import MSD
>>>msd = MSD(22050)
>>>print(msd.calculate_msd(y_ref, y_syn))
2.8384151458740234
```

### Gross Pitch Error (GPE)

```python
>>>from Metrics.gross_pitch_error import GPE
>>>gpe = GPE(22050)
>>>print((gpe.calculate_gpe(y_ref, y_syn))
0.0
```

### Voicing Error Decision (VED)

```python
>>>from Metrics.voicing_decision_error import VDE
>>>vde = VDE(22050)
>>>print(vde(y_ref, y_syn))
0.11494252873563218
```

### Mel Cepstral Distortion (MCD)

```python
>>>from Metrics.mcd import MCD
>>>msd = MSD(22050)
>>>print(mcd.calculate_mcd(y_ref, y_syn))
6.180675506591797
```

### Perceptual Evaluation of Speech Quality (PESQ)

```python
>>>from pesq import pesq
>>>print(pesq(sr_ref, y_ref.numpy()[0], y_syn.numpy()[0], "nb"))
1.9311487674713135
```

### Short Time Objective Intelligibility (STOI)

```python
>>>from pystoi import stoi
>>>print(stoi(y_ref.numpy()[0], y_syn.numpy()[0], sr_ref, extended=False))
0.8629141444017544
```

### Word Error Rate (WER)

```python
>>>from Metrics.word_error_rate import WER
>>>ref = "Hi I love apples"
>>>hyp = "Hi I love oranges"
>>>wer = WER()
>>>print(wer.calculate_wer(ref.split(), hyp.split()))
25.0
```