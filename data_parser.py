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
import sys
import time
# print(os.listdir("./input"))

# Any results you write to the current directory are saved as output.
# -----------------------------------
# Import some audio and visualisation libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt
from shutil import copy2
import sounddevice as sd
import csv
parsed_data_root_path = './parsed_data/'
class_label_filename = parsed_data_root_path+'class_label_indices.csv'
opened_file = open(class_label_filename, 'rt', encoding='utf-8')
class_label_reader = csv.DictReader(opened_file)
c1, c2, *_ = class_label_reader.fieldnames
classes_indices = list()
labels_indices = list()
print(c1,c2)
for row in class_label_reader:
    labels_indices.append(row[c1])
    classes_indices.append(row[c2])
    print(labels_indices[-1], classes_indices[-1])


input_audio_filename = './input/tID-7_speaker-1_trial-2.1.mp3'
input_target_filename = './input/tID-7_speaker-1_trial-2.1_targets.txt'
input_gt_filename = './input/tID-7_speaker-1_trial-2.1_ground_truth.txt'

## ADJUSTABLE PARAMETERS
# over this threshold will be considered as a speech activity
power_threshold = -45
# minimum allowed length for a speech activity (shorter activity will be ignored)
non_silence_threshold = 6
# similar to above but for silence section
silence_threshold = 5
# increase length can remove the hum at the end (22050Hz)
hum_remove_length = 5000


def read_indices_from_csv(filename, encoding='utf-8', verbose=True):
    '''
    Return labels_indices, classes_indices
    As list object containing labels and classes in corresponding index
    '''
    opened_file = open(filename, 'rt', encoding=encoding)
    class_label_reader = csv.DictReader(opened_file)
    c1, c2, *_ = class_label_reader.fieldnames
    classes_indices = list()
    labels_indices = list()
    if verbose:
        print(c1,c2)
    for row in class_label_reader:
        labels_indices.append(row[c1])
        classes_indices.append(row[c2])
        if verbose:
            print(labels_indices[-1], classes_indices[-1])
    return labels_indices, classes_indices


def parse_from_path(path, indices_filename, storage_root_path,
                    power_threshold=-50, silence_threshold=10, 
                    non_silence_threshold=10, hum_remove_length=3300,
                    indices_encoding='utf-8', verbose=True):
    labels_indices, classes_indices = read_indices_from_csv(indices_filename, 
                                                            encoding=indices_encoding, 
                                                            verbose=verbose)
    walk = os.walk(path).__next__()
    for filename in walk[2]:
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            audio_file = os.path.join(path, filename)
            samples, classes, valid = parse_into_samples(audio_file, power_threshold=power_threshold, 
                                                  silence_threshold=silence_threshold, 
                                                  non_silence_threshold=non_silence_threshold, 
                                                  hum_remove_length=hum_remove_length)
            # wait key y or Y to continue
            print('current file:',filename)
            if valid:
                print('press key y or Y to continue, press others to abort.')
                # captured_key = plt.waitforbuttonpress()
                captured_key = input()
                if captured_key == 'y' or captured_key == 'Y':
                    print('saving...')
                    # plt.close('all')
                    save_samples(storage_root_path, samples, classes, labels_indices, classes_indices)
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                    targets_file = audio_file[:-4]+'_targets.txt'
                    if os.path.exists(targets_file):
                        os.remove(targets_file)
                    ground_truth_file = audio_file[:-4]+'_ground_truth.txt'
                    if os.path.exists(ground_truth_file):
                        os.remove(ground_truth_file)
                else:
                    print('unexpected key pressed, aborting...')
                    plt.close('all')
                    exit()
    return


def save_samples(root_path, samples, classes, labels_indices, classes_indices, sampling_rate=22050):
    labels = list()
    file_index = list()
    for i in range(len(labels_indices)):
        dir_path = os.path.join(root_path, str(labels_indices[i]))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_index.append(len([name for name in os.listdir(dir_path) if name.endswith('.wav') or name.endswith('.mp3')]))
    for j in range(len(classes)):
        for i in range(len(classes_indices)):
            if classes_indices[i] == classes[j]:
                labels.append(labels_indices[i])
                break
            if classes_indices[i] == 'unrelated':
                labels.append(labels_indices[i])

        dir_path = os.path.join(root_path, str(labels[-1]))
        file_path = os.path.join(dir_path,str(file_index[int(labels[-1])])+'.wav')
        librosa.output.write_wav(file_path, samples[j], sampling_rate)

    
def parse_into_samples(filename, power_threshold=-50, non_silence_threshold=10, silence_threshold=10, 
                       hum_remove_length=3300, silence_samples=True):
    '''
    Return samples , classes, valid
    As list object containing each sample clips and its corresponding classes
    valid - true for valid results, false for invalid results
    '''
    
    # load audio and remove the hum at the end
    y, sr = librosa.load(filename)
    y = y[:-hum_remove_length]
    # create time axis for plotting
    t = np.arange(0,len(y))/sr

    # get mel spectrogram
    s = librosa.feature.melspectrogram(y=y, sr=sr)
    # scale to dB
    sdB = librosa.power_to_db(s,ref=np.max)

    # Import target words list and ground truth
    if filename.endswith('.wav'):
        # input_target_filename = filename.replace('.wav','_targets.txt')
        input_gt_filename = filename.replace('.wav','_ground_truth.txt')
    elif filename.endswith('.mp3'):
        # input_target_filename = filename.replace('.mp3','_targets.txt')
        input_gt_filename = filename.replace('.mp3','_ground_truth.txt')
    else:
        print('Invalid filename, must end with .wav or .mp3 . Aborting...')
        exit()
    # targets = np.genfromtxt(input_target_filename, delimiter=' ', dtype=str)
    groundtruth = np.genfromtxt(input_gt_filename, delimiter=' ', dtype=str)

    # power in frequencies below 512 Hz
    p512 = (sdB[0:31,:]).mean(axis=0)
    # threshold to find the transition of sound and silence
    mask=np.where(p512>power_threshold,1,0)
    mask2 = np.zeros(len(mask))
    # identify transition from silence to non-silence
    one_to_zero=[]
    zero_to_one=[]
    # init an one to zero transition at beginning
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
    # finalise with a zero to one transition at end
    zero_to_one.append(len(mask)-1)

    # parse out silence and non-silence parts
    silence = np.array(list(zip(one_to_zero, zero_to_one)))
    non_silence = np.array(list(zip(zero_to_one[:-1], one_to_zero[1:])))

    middle_positions = list()     
    
    # middle point of each section of sound
    for x in non_silence:
        mask2[x[0]:x[1]]=np.ones(x[1]-x[0])
        middle_positions.append(x.mean())

    # split these intervals of silence
    pauses = [x.mean() for x in silence]
    # convert to nearest integers array
    pauses = np.array(pauses).astype(int)
    samples = list()
    classes = list()
    number_samples = len(middle_positions)

    # check if the non-silence clips match the number of words
    print('expect:',len(groundtruth),'found:',number_samples)
    if number_samples != len(groundtruth):
        return samples, classes, False
    # parse out samples and its class
    index_position = 0
    silence_margins = list()
    silence_all = np.empty((0))
    for i in range(number_samples):
        left_margin = int(max(non_silence[i][0]-silence_threshold, silence[i].mean()) * t.max() / sdB.shape[1] * sr)
        right_margin = int(min(non_silence[i][1]+silence_threshold, silence[i+1].mean()) * t.max() / sdB.shape[1] * sr)
        samples.append(y[left_margin:right_margin])
        classes.append(groundtruth[i])
        if silence_samples and left_margin-index_position>20:
            silence_all = np.concatenate((silence_all, y[index_position:left_margin]))
            samples.append(y[index_position:left_margin])
            classes.append('silence')
            index_position = right_margin
            silence_margins.append(len(silence_all)-1)
    if silence_samples:
        silence_all = np.concatenate((silence_all, y[index_position:]))

    # plot partition results for validation
    f1 = plt.subplots(2,1,sharex=True,figsize=(6,6))
    ax1=plt.subplot(3,1,1)
    plt.plot(p512,'m')
    plt.plot(pauses,p512[pauses],'r.')
    plt.axhline(y=power_threshold, color='g',linestyle=':')
    plt.title("Mean mel-binned power below 512Hz")
    plt.ylabel('dB')

    ax2=plt.subplot(3,1,2)
    mask=np.where(p512>power_threshold,1,0)
    plt.plot(mask2,'g')
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
    number_samples = len(samples)

    # ff = plt.subplots(number_samples,1)
    # for i in range(number_samples):
    #     ax = plt.subplot(number_samples,1,i+1)
    #     plt.plot(samples[i])

    plt.show(block=False)
    for i in range(len(classes)):
        print('#',i,'label:',classes[i])
        time.sleep(0.5)
        sd.play(samples[i], sr, blocking=True)
        time.sleep(0.5)
    
    return samples, classes, True


parse_from_path('./input/',class_label_filename,parsed_data_root_path,power_threshold=power_threshold, silence_threshold=silence_threshold, non_silence_threshold=non_silence_threshold, hum_remove_length=hum_remove_length)


