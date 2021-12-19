import soundfile as sf
from pesq import pesq #install using pip install pesq

if __name__=="__main__":
    ## This implementation of PESQ supports both narrow band and wide band PESQ calculations
    ref, sr = sf.read("../Testset/clean/sp01.wav") #input the clean speech sample
    deg, sr = sf.read("../Testset/noisy/babble/sp01_babble_sn10.wav") # input the corresponding noisy speech sample
    score = pesq(sr, ref, deg, "nb")
    score = pesq(sr, ref, deg, "wb") #no wide band mode if fs = 8000
    
    print(score)