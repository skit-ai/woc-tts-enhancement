import torchaudio
from pystoi import stoi

if __name__== "__main__":
	y_ref, sr_ref = torchaudio.load('../Testset/clean/sp01.wav')
	y_ref, sr_syn = torchaudio.load('../Testset/noisy/babble/sp01_babble_sn10.wav')
	stoi_score = stoi(y_ref.numpy()[0], y_syn.numpy()[0], sr_ref, extended=False)
	print(stoi_score)
