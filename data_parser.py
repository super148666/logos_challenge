# Some info re. this iPython notebook:
# -----------------------------------
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("./input"))

# Any results you write to the current directory are saved as output.
# -----------------------------------
# Import some audio and visualisation libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt

input_audio_filename = './input/tID-7_speaker-1_trial-2.1.mp3'
input_target_filename = './input/tID-7_speaker-1_trial-2.1_targets.txt'
input_gt_filename = './input/tID-7_speaker-1_trial-2.1_ground_truth.txt'

power_threshold = -50
non_silence_threshold = 10
silence_threshold = 5
# 1 - Import and load the wav file (using librosa)
y, sr = librosa.load(input_audio_filename)
print('frequency:',sr,'Hz')
max_clip_length = sr * 2
y = y[:-3300]
t = np.arange(0,len(y))/sr

# # Have a look!
# f0 = plt.figure(figsize=(12,4))
# plt.plot(t,y)
# plt.title(input_audio_filename)
# plt.xlabel('time (s)')
# plt.ylabel('amplitude')

# Compute mel-scaled spectrogram, and scale to dB
s = librosa.feature.melspectrogram(y=y, sr=sr)
sdB = librosa.power_to_db(s,ref=np.max)

# Import target words list and ground truth
targets = np.genfromtxt(input_target_filename, delimiter=' ', dtype=str)
groundtruth = np.genfromtxt(input_gt_filename, delimiter=' ', dtype=str)

# 2 - Estimate word locations
# > Take a look at the power in frequencies below 512 Hz, covering the resonant freqs of human speech
p512 = (sdB[0:31,:]).mean(axis=0)
# > threshold this mean above -45 dB, to estimate where speech is found
mask=np.where(p512>power_threshold,1,0)
mask3 = np.zeros(len(mask))
# A crude partitioning algorithm
one_to_zero=[]
zero_to_one=[]
one_to_zero.append(0)
for i in range(len(mask)-1):
    if (mask[i] == 1) & (mask[i+1] == 0):
        if (len(zero_to_one) == 0):
            one_to_zero.append(i)
        elif ((i-zero_to_one[-1])>non_silence_threshold):
            one_to_zero.append(i)
        else:
            del zero_to_one[-1]
        
    elif (mask[i] == 0) & (mask[i+1] == 1):
        if (len(one_to_zero) == 0):
            zero_to_one.append(i)
        elif ((i-one_to_zero[-1])>silence_threshold):
            zero_to_one.append(i)
        else:
            del one_to_zero[-1]
zero_to_one.append(len(mask)-1)
# > generates the locations where 1s become 0s and vice versa
silence = np.array(list(zip(one_to_zero, zero_to_one)))
non_silence = np.array(list(zip(zero_to_one[:-1], one_to_zero[1:])))

for x in non_silence:
    mask3[x[0]:x[1]]=np.ones(x[1]-x[0])
# > split these intervals of silence
pauses = [x.mean() for x in silence]
# > convert to nearest integers array
pauses = np.array(pauses).astype(int)


# Have a look!
f1 = plt.subplots(2,1,sharex=True,figsize=(12,12))
ax1=plt.subplot(3,1,1)
plt.plot(p512,'m')
plt.plot(pauses,p512[pauses],'r.')
plt.title("Mean mel-binned power below 512Hz")
plt.ylabel('dB')

ax2=plt.subplot(3,1,2)
mask=np.where(p512>power_threshold,1,0)
plt.plot(mask3,'g')
plt.plot(pauses,mask[pauses],'r.')
plt.title("Speech indicator boolean")
plt.ylabel('boolean')

ax2=plt.subplot(3,1,3)
plt.plot(t,y)
for xc in pauses*t.max()/sdB.shape[1]:
    plt.axvline(x=xc,color='r',linestyle=':')
plt.title("Partitioned audio stream")
plt.xlabel('time (s)')
plt.ylabel('amplitude')

plt.show()

# # 3 - Now that we have partitioned the audiostream, let's have a look at a couple words up close, in their captured
# # representation as air pressure perturbation amplitudes

# # Plot some words
# # > compute indices for down-sampled mel-spectrum and full time domain
# word_times_m = list(zip(pauses,pauses[1:]))
# word_times_t = np.array(word_times_m)*t.max()/sdB.shape[1]


# # Prepare figure
# f2, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4), sharey=True)
# ax1 = plt.subplot(1,2,1)
# plt.ylabel('amplitude')
# plt.xlabel('time (s)')

# plt.plot(t,y)
# # Zoom in on first word
# ax1.set_xlim([word_times_t[0,0], word_times_t[0,1]])
# # Adjust plot 
# plt.title("Waveform of '{}'".format(groundtruth[0]))
# plt.tight_layout()

# ax2 = plt.subplot(1,2,2)
# plt.plot(t,y)
# # Zoom in on second word
# ax2.set_xlim([word_times_t[1,0], word_times_t[1,1]])
# # Adjust plot 
# plt.title("Waveform of '{}'".format(groundtruth[1]))
# plt.tight_layout()
# plt.xlabel('time (s)')

# plt.show()

# # 4 - Now let's have a look at what these words look like in the mel-spectrogram representation
# # The mel-spectrogram is a graphical representation of the power spectral densities of the waveform at successive timebins.
# # The 'mel' prefix refers to a scaling of the frequency bins into roughly equal energy -- ie, more bins at lower frequencies,
# # and fewer bins at higher ones.


# # Prepare figure
# f3, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4), sharey=True)

# # Plot some words
# ax1 = plt.subplot(1,2,1)
# librosa.display.specshow(sdB, y_axis='mel', fmax=8000, x_axis='time')
# # Zoom in on first word
# ax1.set_xlim([word_times_t[0,0], word_times_t[0,1]])
# # Adjust plot 
# plt.colorbar(format='%+2.0f dB')
# plt.title("Mel spectrogram for '{}'".format(groundtruth[0]))
# plt.tight_layout()

# ax2 = plt.subplot(1,2,2)
# librosa.display.specshow(sdB, y_axis='mel', fmax=8000, x_axis='time')
# # Zoom in on second word
# ax2.set_xlim([word_times_t[1,0], word_times_t[1,1]])
# # Adjust plot 
# plt.colorbar(format='%+2.0f dB')
# plt.title("Mel spectrogram for '{}'".format(groundtruth[1]))
# plt.tight_layout()

# plt.show()