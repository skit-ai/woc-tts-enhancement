import scipy
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
plt.style.use('ggplot')
import pandas as pd


def output_duration(length):
	hours = length // 3600 # calculate in hours
	length %= 3600
	mins = length // 60 # calculate in minutes
	length %= 60
	seconds = length # calculate in seconds
	return hours, mins, seconds


gen_durations = []

for dirname, _, filenames in os.walk("../data/generated_audio/wavs"):
    for filename in filenames:
        sample_rate, data = wavfile.read(os.path.join(dirname, filename))
        len_data = len(data) # holds length of the numpy array
        t = len_data / sample_rate # returns duration but in floats
        hours, mins, seconds = output_duration(int(t))
        gen_durations.append(seconds)

gen_durations = np.array(gen_durations)
n, bins, patches = plt.hist(gen_durations)
plt.show()
plt.savefig('gen.png')

train_durations = []

for dirname, _, filenames in os.walk("../data/training_audio/wavs"):
    for filename in filenames:
        sample_rate, data = wavfile.read(os.path.join(dirname, filename))
        len_data = len(data) # holds length of the numpy array
        t = len_data / sample_rate # returns duration but in floats
        hours, mins, seconds = output_duration(int(t))
        train_durations.append(seconds)

train_durations = np.array(train_durations)
n, bins, patches = plt.hist(train_durations)
plt.show()
plt.savefig('train.png')

diff_durations = []

file_train = []
file_gen = []

t_duration = []
g_duration = []

diff_data = pd.DataFrame()

for dirname_train, _, filenames_train in os.walk("../data/training_audio/"):
    for filename_train in filenames_train:
        for dirname_gen, _, filenames_gen in os.walk("../data/generated_audio"):
            for filename_gen in filenames_gen:
                if filename_train in filename_gen:
                    t = AudioSegment.from_wav(os.path.join(dirname_train, filename_train))
                    g = AudioSegment.from_wav(os.path.join(dirname_gen, filename_gen))
                    
                    tmili = len(t)
                    gmili = len(g)
                    
                    t_duration.append(tmili)
                    g_duration.append(gmili)
                    file_train.append(filename_train)
                    file_gen.append(filename_gen)
                    diff_durations.append(tmili-gmili)

diff_data["Train_filename"] = file_train
diff_data["Gen_filename"] = file_gen
diff_data["Train_duration"] = t_duration
diff_data["Gen_duration"] = g_duration
diff_data["Difference"] = diff_durations

diff_data.to_csv('tts_data.csv')
