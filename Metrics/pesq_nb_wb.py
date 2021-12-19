import torchaudio
from pesq import pesq #install using pip install pesq

if __name__=="__main__":
    ## This implementation of PESQ supports both narrow band and wide band PESQ calculations
    y_ref, sr = torchaudio.load("../Testset/clean/sp01.wav") #input the clean speech sample
    y_syn, sr = torchaudio.load("../Testset/noisy/babble/sp01_babble_sn10.wav") # input the corresponding noisy speech sample

    y_ref = y_ref.numpy()[0]
    y_syn = y_syn.numpy()[0]
    score = pesq(sr, y_ref, y_syn, "nb")
    #score = pesq(sr, y_ref, y_syn, "wb") #no wide band mode if fs = 8000
    
    print(score)