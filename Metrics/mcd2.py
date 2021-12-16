import torch
import torchaudio
import numpy as np

class MCDMetric(torch.nn.Module):
    def __init__(self, sr=16000):
        super().__init__()
        # 5 ms windows, taken from
        # "SYNTHESIZER VOICE QUALITY OF NEW LANGUAGES CALIBRATED WITH MEAN MEL CEPSTRAL 
        # DISTORTION", Kominek et al.
        
        # See also https://dsp.stackexchange.com/questions/56391/mel-cepstral-distortion

        hop = n_fft = ws = sr * 5 // 100 
        n_mels = 80

        mel_kwargs = dict(hop_length=hop, n_mels=n_mels, n_fft=n_fft, window_fn=torch.hann_window)

        self.mfcc_transform = torchaudio.transforms.MFCC(sr,
                                                         n_mfcc=13,
                                                         log_mels=True,
                                                         melkwargs=mel_kwargs)
        self.scaler = 10.0 / np.log(10.0) * np.sqrt(2)

    def forward(self, original, restored):
        assert original.size() == restored.size()
        original_mfcc = self.mfcc_transform(original)
        restored_mfcc = self.mfcc_transform(restored)

        distortion = (original_mfcc - restored_mfcc)[:, :, 1:]  # cut the first band
        distortion = distortion.pow(2.0).sum(dim=-1).sqrt().mean(dim=-1) * self.scaler

        return distortion

if __name__ == "__main__":
    path1 = '/root/shangeth/ModularTTS/audio_samples/b-3_2350-2433_2379.wav'
    # path2 = '/root/shangeth/ModularTTS/audio_samples/b-4_1601-1706_1703.wav'

    mcd = MCDMetric(22050)
    # print(gpe.calculate_gpe_path(path1, path2))

    y_ref, sr_ref = torchaudio.load(path1)
    y_syn = y_ref + torch.randn(y_ref.size()) * 0.001

    # print(y_ref.size(), y_syn.size())
    print(mcd(y_ref, y_syn))
