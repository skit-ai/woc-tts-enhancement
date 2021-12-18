import soundfile as sf
from pypesq import pesq #install using pip install pypesq

if __name__=="__main__":
    ref, sr = sf.read("../Testset/clean/sp01.wav") #input the clean speech sample
    
    deg, sr = sf.read("../Testset/noisy/babble/sp01_babble_sn10.wav") # input the corresponding noisy speech sample
    score = pesq(ref, deg, sr)
    print(score)
    
    deg, sr = sf.read("../Testset/noisy/station/sp01_station_sn15.wav") 
    score = pesq(ref, deg, sr)
    print(score)