# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:21:59 2018

@author: jake
"""
import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt

pathdir={"LOGOS":"C:/Users/jaked/Desktop/speech recognition/LOGOS datasets/LOGOS exemplar data/LOGOS exemplar training set",
          "simulated":"C:/Users/jaked/Desktop/speech recognition/LOGOS datasets/simulated data/simulated_data training set",
          "parsed_data":"C:/Users/jaked/Desktop/speech recognition/parsed_data"
        }

def load_data(path):
    X=[]
    Y=[]
    for file_name in os.listdir(path):
        #exclude slience folder
        if os.path.isdir(path+'/'+file_name) and file_name != "15":       
            for audio_name in os.listdir(path+'/'+file_name):
                audio_path = path+'/'+file_name+'/'+audio_name
                x, sr = librosa.load(path = audio_path, sr = 22050, mono = True)   
                X.append(x)
                Y.append(int(file_name)) 
    #change into numpy array
    Y = np.asarray(Y)
    return X , Y, sr

def resample(X, sr, method):
    data_length = []
    for i in range (len(X)):
        data_length.append(X[i].shape)
    if method == 'median':
        target_length = np.median(data_length)
    elif method == 'min':
        target_length = np.min(data_length)
    #resample to adjust the data_length
    new_sr = []
    new_X = np.zeros((len(X), target_length))
    for i, audio in enumerate(X):    
        target_sr = np.round(target_length/(data_length[i][0]/sr))
        temp = librosa.core.resample(audio, sr, target_sr)
        new_X[i] = temp[:target_length]
        new_sr.append(target_sr)
        
    return new_X, new_sr       
              
def feature_extract(X, nfft, nb):
    mfcc = np.zeros((X.shape[0], nb, int(4*np.round(X.shape[1]/nfft)))) 
    for i, audio in enumerate(X):
        #mel spectrum
        spec = librosa.feature.melspectrogram(y=audio, sr=22050,n_fft = nfft, hop_length = int(nfft/4), power =1)
        #mfcc
        feat = librosa.feature.mfcc( sr=22050, S=librosa.core.power_to_db(spec), n_mfcc=nb, dct_type=2, norm='ortho')
        mfcc[i] = feat
    return mfcc

def plot_feat(feat):
    #plot 2D mfcc
    if len(feat.shape) == 2:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(feat, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
    #plot 1D mfcc
    elif len(feat.shape) == 1:
        plt.plot(feat)
    
    

#train model        
if __name__ == "__main__":
    #load time domain signal
    X, Y, sr = load_data(pathdir["parsed_data"])
    #resample into same length
    #min or meidan length
    new_X, new_sr = resample(X, sr, 'min')
    #feature extraction
    #mbe or mfcc feature
    mfcc_2d = feature_extract(X= new_X, nfft = 2048,nb =  20)
    #plot out one of them
    plot_feat(mfcc_2d[0])
    #reshape into 1D
    mfcc_1d = mfcc_2d.reshape(mfcc_2d.shape[0], mfcc_2d.shape[1]*mfcc_2d.shape[2])
    plot_feat(mfcc_1d[31])
    
    #pca
    
    #train model
    


    
    
    