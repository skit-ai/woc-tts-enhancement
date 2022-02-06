import numpy as np
import wave
import scipy.integrate as integ
from scipy.special import iv

import numpy as np

class Kalman(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics!")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x0 = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x0 = np.dot(self.F, self.x0) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x0

    def update(self, z):
        y = z - np.dot(self.H, self.x0)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x0 = self.x0 + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

    def Kalman_path(self, path1, path2):
        f = wave.open(path1)
        params = f.getparams()
        nchannels, sampwidth, fs, nframes = params[:4]
        # read wave data
        str_data = f.readframes(nframes)
        # close .wav file
        f.close()
        # convert waveform data to an array
        x = np.fromstring(str_data, dtype=np.short) ## x is denormalized here

        filtered = []

        for z in x:
            filtered.append(np.dot(self.H,  self.predict())[0])
            self.update(z)
        
        wf = wave.open(path2, 'wb')
        # setting parameters
        wf.setparams(params)
        # set waveform file .tostring()Convert array to data
        wave_data = np.asarray(filtered, dtype=np.short)
        wf.writeframes(wave_data.tostring())
        # close wave file
        wf.close()

if __name__ == '__main__':
    path1 = "../docs/audio/sp15_station_sn5.wav"
    path2 = "../docs/audio/kalman_filtered_sp15_station_sn5.wav"
    dt = 1.0/60
    
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)
    
    kalman_filter = Kalman(F = F, H = H, R = R, Q = Q)
    kalman_filter.Kalman_path(path1, path2)
    print("Audio denoised!")
