import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from skimage.transform import resize

def detect_key(audio_path):
    # referenced from https://medium.com/@oluyaled/detecting-musical-key-from-audio-using-chroma-feature-in-python-72850c0ae4b1
    """input: audio path
       output: key of the original wav file and chromagram
       detect the key by finding the maximum value of mean chroma"""
    y, sr = librosa.load(audio_path)
    # test path "F:/FYP/test_shs100k_small_sample/3/ffmpeg_Amy Nuttall - Greensleeves.wav"
    # Compute the Chroma Short-Time Fourier Transform (chroma_stft)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chromagram, sr=sr, y_axis='cqt_note', x_axis='time', ax=ax)
    ax.set_title('Chromagram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

    # Calculate the mean chroma feature across time
    mean_chroma = np.mean(chromagram, axis=1)

    # Define the mapping of chroma features to keys
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Find the key by selecting the maximum chroma feature
    estimated_key_index = np.argmax(mean_chroma)
    estimated_key = chroma_to_key[estimated_key_index]

    # Print the detected key
    print("Detected Key:", estimated_key)
    return estimated_key

def shift_to_C_key(audio_path,save_shifted_wav = True):
    """input: audio path, save_shifted_wav (bool value to specify to save shifted wav, default = True
       output: None of shifted wav"""
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key = detect_key(audio_path)

    if(estimated_key != 'C'):
        #shift to C key if the detected key is not C key
        shift_distance = chroma_to_key.index(estimated_key)
        y, sr = librosa.load(audio_path)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-shift_distance)
        if(save_shifted_wav):
            sf.write(audio_path[:-4]+"shifted.wav",y_shifted,sr)
            print("wav shifted to C key")


def compute_parsons_code(input_audio,print_melody=False):

    """input: input_audio (wav file which only contains melody)
       output: parsons code with 1 indicating upward, 0 indicating rest, -1 indicating downward melodic countour

       note for parsons code:
       u = "up", for when the note is higher than the previous note, 1 in our representation
       d = "down", for when the note is lower than the previous note, 0  in our representation
       r = "repeat", for when the note has the same pitch as the previous note. -1 in our representation
       for convenienve, the first element is set to be 0"""
    "F:/FYP/test_shs100k_small_sample/3/vocals_ffmpeg_Amy Nuttall - Greensleeves_shifted.wav"
    y, sr = librosa.load(input_audio)
    cqt_threshold = 2000
    cqt = librosa.cqt(y=y, sr=sr)
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    cqt = resize(cqt, (12, cqt_threshold))
    print(cqt.shape)
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    melody_line = np.argmax(cqt,axis=0)
    print(melody_line)
    print(melody_line.shape)
    print([cqt[i][1024] for i in range(12)])
    print(melody_line[1024])
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


    print(parsons_code)
    #print(22//12)

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








compute_parsons_code(input_audio="C:\\FYP_data_test\\wav_test\\1\_ Silent Night _ Trace Adkins -The King´s Gift feat. Kevin Costner & Lily Costner.wav")






