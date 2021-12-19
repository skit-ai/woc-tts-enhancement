import soundfile as sf
from pystoi import stoi #pip install pystoi

if __name__== "__main__":
	clean, fs = sf.read('../Testset/clean/sp01.wav')
	denoised, fs = sf.read('../Testset/noisy/babble/sp01_babble_sn10.wav')
	# Clean and den should have the same length, and be 1-Dimensional
	d = stoi(clean, denoised, fs, extended=False)
	print(d)
