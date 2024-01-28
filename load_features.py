#import deepdish as dd
import os
import pickle
#import glob
#import csv
import numpy as np
#import torch
#from sklearn.model_selection import train_test_split
#import pandas as pd
#from torch.utils.data import Dataset, DataLoader
import librosa
#import scipy.io as sio
#import soundfile as sf
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import libfmp.c3
import random
from skimage.transform import resize
import matplotlib.pyplot as plt



def load_data(pre_extracted_data_path = "",audio_dir="F:\FYP\\test_shs100k_small_sample",extracted_cqt = False,down_sample = False,slice_cqt = True,all_data_saved_path = ""):
    '''args: pre_extracted_data_path (path to the pre_extracted features in npy format, optional)
             audio_dir: path to the root directory where all audio folders are stored
             extracted_cqt: if true then the code will load .npy instead of computing cqt from wav file
             down_sample: downsample the cqt using the function provided by libfmp
             slice_cqt: if true then will slice the cqt into exerpts as the way stated in mirex paper
             all_data_saved: bool, if true, will load the .npy file that store all of the small.npy files directly, which is automatically generated after running load_data once
       output: the nparray which contain all input features, can be passed to split_training_data to split them'''
    if(all_data_saved_path != ""):
        if all_data_saved_path.endswith(".npy"):
            if os.path.getsize(all_data_saved_path) > 0:
                print("loading .npy files storing all training data in" + str(all_data_saved_path))
                all_data = np.load(all_data_saved_path, allow_pickle=True)
            else:
                pass
        else:
            raise IOError("Incorrect file type!")
        return all_data


    if(pre_extracted_data_path):
        print("Data loaded from specified directory")
        return np.load(pre_extracted_data_path,allow_pickle=True)

    all_data = np.array([])
    temp_class_list = os.listdir(audio_dir)
    class_list = []
    for i in range (len(temp_class_list)):
        try:
            if os.listdir(audio_dir + "/" + temp_class_list[i]) != []:
                class_list.append(temp_class_list[i])
        except:
            pass
    #print(class_list)
    #class_list.remove("0")
    class_list_len = len(class_list)
    class_label = [i for i in range(class_list_len)]
    mapping = dict(zip(class_list, class_label))
    #print(mapping)
    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(mapping, f)
        f.close()
    print("mapping saved into saved_dictionary.pkl, " + str(class_list_len) + " classes in total", )
    count =0
    threshold_cqt_len = 10000
    for classes in class_list:
        #print(os.listdir(audio_dir + "/" + classes))
        wav_dir = audio_dir + "/" + classes + "/"
        try:
            for file in os.listdir(audio_dir + "/" + classes):

                    if extracted_cqt:
                        if file.endswith(".npy"):
                            if os.path.getsize(audio_dir + "/" + classes+ "/" + file) > 0:
                                cqt = np.load(audio_dir + "/" + classes+ "/" +file,allow_pickle=True)

                                if(np.isnan(cqt).any()):
                                    print("This file contains NaN:", file)
                                    pass
                            else:
                                pass
                        else:
                            raise IOError("Incorrect file type!")
                    else:
                        cqt = extract_cqt(audio_dir + "/" + classes+ "/" +file)
                    #print(cqt.shape)
                    if(int(cqt.shape[1])>threshold_cqt_len):
                        cqt = cqt[:,:threshold_cqt_len]
                    else:
                        cqt = np.pad(cqt,((0, 0), (0, threshold_cqt_len-int(cqt.shape[1]))),'constant')

                    if down_sample:
                        #decrease the time axis by 20 times, now cqt shape is (84,1000)
                        cqt, _ = libfmp.c3.c3s1_post_processing.smooth_downsample_feature_sequence(cqt,22050,down_sampling=5)
                        #print(cqt.shape)
                    #normalization using Per-channel energy normalization (PCEN)
                    #cqt = librosa.pcen(cqt)
                    #print(cqt.shape)
                    count += 1
                    new_row = np.array([mapping.get(classes), file, cqt.astype(np.float32)], dtype=object)
                    all_data = np.append(all_data, new_row, axis=0)
                    #print(cqt.shape)
                    #print(new_row[2].shape)
                    #print(new_row)
                    if (count %1000 ==0):
                        print("Loading Data item: ",count)
        except IOError:
            pass
        except PermissionError:
            pass
        except EOFError:
            pass
        except pickle.UnpicklingError:
            pass
        except pickle.UnpicklingError:
            pass



    all_data = all_data.reshape((count, 3))

    if(slice_cqt):
        count =0
        print("slicing cqt...")
        sliced_all_data = np.array([])
        for i in range(len(all_data)):
            original_cqt = all_data[i][2]
            slicing_range = [200,300,400]
            for ranges in slicing_range:
                cqt_begin = random.randint(0,threshold_cqt_len-ranges)
                sliced_cqt = np.concatenate((np.zeros((84,cqt_begin)), original_cqt[:,cqt_begin:cqt_begin+ranges],np.zeros((84,threshold_cqt_len-cqt_begin-ranges))), axis=1)
                new_row = np.array([all_data[i][0], all_data[i][1], sliced_cqt.astype(np.float32)], dtype=object)
                sliced_all_data = np.append(sliced_all_data, new_row, axis=0)
                count +=1

        sliced_all_data = sliced_all_data.reshape((count, 3))
        with open('111_cqt_org_alldata.npy', 'wb') as f:
            np.save(f, all_data)
            f.close()
        return sliced_all_data


    class_dict = {}
    print("saving data")
    with open('new_all_melody_cqt_term1.npy', 'wb') as f:
        np.save(f, all_data)
        f.close()
    return all_data

def extract_cqt(wav_dir):
    '''extract cqt for one file, set hop len to 2048 to decrease size of cqt, not sure if it will decrease frequency resolution'''
    y, sr = librosa.load(wav_dir)
    cqt = librosa.cqt(y,hop_length=1024)
    return cqt

def split_training_data(input_data):
    """input data: the nparray which contains all input features, which is the output of load_data"""
    print("splitting training data and test data")
    loaded_data = input_data
    print("loaded data shape:", loaded_data.shape)

    # Set the random seed for reproducibility
    np.random.seed(42)
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(loaded_data, test_size=0.1, random_state=42)

    print("train_data.shape", train_data.shape)
    print("test_data.shape", test_data.shape)

    #produce x and y data for training set and test set
    x_train = np.array([train_data[i][2] for i in range(int(train_data.shape[0]))])
    y_train = np.array([train_data[i][0] for i in range(int(train_data.shape[0]))])

    x_test = np.array([test_data[i][2] for i in range(int(test_data.shape[0]))])
    y_test = np.array([test_data[i][0] for i in range(int(test_data.shape[0]))])

    return x_train,y_train,x_test,y_test

def load_parsons(audio_dir="F:\FYP\\test_shs100k_small_sample",extracted_cqt = False,all_data_saved_path = ""):
    '''args: pre_extracted_data_path (path to the pre_extracted features in npy format, optional)
             audio_dir: path to the root directory where all audio folders are stored
             extracted_cqt: if true then the code will load .npy instead of computing cqt from wav file
             down_sample: downsample the cqt using the function provided by libfmp
             slice_cqt: if true then will slice the cqt into exerpts as the way stated in mirex paper
             all_data_saved: bool, if true, will load the .npy file that store all of the small.npy files directly, which is automatically generated after running load_data once
       output: the nparray which contain all input features, can be passed to split_training_data to split them'''
    if(all_data_saved_path != ""):
        if all_data_saved_path.endswith(".npy"):
            if os.path.getsize(all_data_saved_path) > 0:
                print("loading .npy files storing all training data...")
                all_data = np.load(all_data_saved_path, allow_pickle=True)
            else:
                pass
        else:
            raise IOError("Incorrect file type!")
        return all_data

    all_data = np.array([])
    temp_class_list = os.listdir(audio_dir)
    class_list = []
    for i in range (len(temp_class_list)):
        try:
            if os.listdir(audio_dir + "/" + temp_class_list[i]) != []:
                class_list.append(temp_class_list[i])
        except:
            pass
    #print(class_list)
    #class_list.remove("0")
    class_list_len = len(class_list)
    class_label = [i for i in range(class_list_len)]
    mapping = dict(zip(class_list, class_label))
    print(mapping)
    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(mapping, f)
        f.close()
    print("mapping saved into saved_dictionary.pkl, " + str(class_list_len) + " classes in total", )
    count =0
    for classes in class_list:
        #print(os.listdir(audio_dir + "/" + classes))
        wav_dir = audio_dir + "/" + classes + "/"
        try:
            for file in os.listdir(audio_dir + "/" + classes):
                    if extracted_cqt:
                        if file.endswith(".npy"):
                            if os.path.getsize(audio_dir + "/" + classes+ "/" + file) > 0:
                                cqt = np.load(audio_dir + "/" + classes+ "/" +file,allow_pickle=True)
                                parsons = compute_parsons_code(cqt)

                                if(np.isnan(parsons).any()):
                                    print("This file contains NaN:", file)
                                    pass
                            else:
                                pass
                        else:
                            raise IOError("Incorrect file type!")
                    else:
                        parsons = compute_parsons_code(audio_dir + "/" + classes+ "/" +file,extracted_cqt=False)
                    np_parsons = np.array(parsons)
                    np_parsons = np_parsons[np.newaxis , ...]
                    count += 1
                    new_row = np.array([mapping.get(classes), file, np_parsons], dtype=object)
                    all_data = np.append(all_data, new_row, axis=0)
                    #print(cqt.shape)
                    #print(new_row[2].shape)
                    #print(new_row)
                    if (count %1000 ==0):
                        print("Loading Data item: ",count)
        except IOError:
            pass
        except PermissionError:
            pass
        except EOFError:
            pass
        except pickle.UnpicklingError:
            pass
        except pickle.UnpicklingError:
            pass
    all_data = all_data.reshape((count, 3))


    print("saving data")
    with open('parsons.npy', 'wb') as f:
        np.save(f, all_data)
        f.close()
    return all_data

def compute_parsons_code(input,print_melody=False,extracted_cqt = True):

    """input: input_audio (wav file which only contains melody)
       output: parsons code with 1 indicating upward, 0 indicating rest, -1 indicating downward melodic countour

       note for parsons code:
       u = "up", for when the note is higher than the previous note, 1 in our representation
       d = "down", for when the note is lower than the previous note, 0  in our representation
       r = "repeat", for when the note has the same pitch as the previous note. -1 in our representation
       for convenience, the first element is set to be 0"""
    "F:/FYP/test_shs100k_small_sample/3/vocals_ffmpeg_Amy Nuttall - Greensleeves_shifted.wav"
    cqt_threshold = 2000
    if not extracted_cqt:
        y, sr = librosa.load(input)
        cqt = librosa.cqt(y=y, sr=sr)
    else:
        cqt = input
    #cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    cqt = resize(cqt, (84, cqt_threshold))
    librosa.display.specshow(cqt)
    #print(cqt.shape)
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    melody_line = np.argmax(cqt,axis=0)

    plt.plot(melody_line)
    plt.show()

    #print(melody_line)
    #print(melody_line.shape)
    #print([cqt[i][1024] for i in range(12)])
    #print(melody_line[1024])
    parsons_code=[]
    melody_length = len(melody_line)
    for i in range(melody_length-1):
        if(i==0):
            parsons_code.append(0)
        else:
            if (melody_line[i] == melody_line[i+1]):
                parsons_code.append(0)
            elif (melody_line[i] < melody_line[i+1]):
                parsons_code.append(1)
            else:
                parsons_code.append(-1)
    #print(parsons_code)


    if print_melody:
        melody_wav = np.array([])
        for i in range(melody_length):
            pitch_height = melody_line[i]//12
            pitch_class = chroma_to_key[melody_line[i]%12]
            note_hz = librosa.note_to_hz(str(pitch_class+str(pitch_height+4)))
            #print(librosa.tone(note_hz, sr=22050, length=22050/16).shape)
            melody_wav = np.concatenate((melody_wav, librosa.tone(note_hz, sr=22050, length=11025 )), axis=None)
        sf.write("test_melody_use_chromacqt.wav",melody_wav,22050)

    return parsons_code

#modify to specify the path which stores the pre-extracted features
#load_features('F:/FYP/da-tacos_coveranalysis_subset_single_files/')
#extract_cqt_test()
#load_data()
#split_training_data()
compute_parsons_code(np.load("F:\FYP\\tim_data\\train_melody01\\1147\\Ayo - I Want You Back.npy"))