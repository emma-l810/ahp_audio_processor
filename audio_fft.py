import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np, scipy
import os, os.path

"""
    Resources used:
    https://librosa.org/doc/latest/tutorial.html
    http://librosa.org/doc/main/generated/librosa.stft.html
    https://towardsdatascience.com/all-you-need-to-know-to-start-speech-processing-with-deep-learning-102c916edf62#:~:text=Hop%20length%20is%20the%20length,portion%20of%20the%20window%20length.

    Sounds:
    Violin - https://freesound.org/people/Carlos_Vaquero/sounds/153592/
    Flute - https://freesound.org/people/MTG/sounds/354672/
"""

def load_audio(filename):
    ### LOAD W LIBROSA ###
    # filename = input('File Path: ') # eg: './audio_files/sample_fs_violin_d4.wav'
    # loads and decodes audio file into time series and sample rate
    y, sr = librosa.load(filename)

    print(f'waveform: {y}')
    print(f'sample rate: {sr}')
    print(f'shape: {y.shape}')

    return y, sr


def get_graphs(y, sr, title):
    ### WAVEPLOT ###
    plt.figure(figsize = (15,5))
    librosa.display.waveshow(y, sr=22050)
    #plt.show()

    ### SHORT FOURIER TRANSFORM ###
    # stft = np.abs(librosa.stft(y))
    # no librosa -> switch to scipy
    Y = np.fft.fft(y[20000:24096])
    Y_mag = np.absolute(Y) # spectral magnitude
    f = np.linspace(0, sr/2, len(Y_mag)) #frequency var
    plt.figure(figsize=(14,5))
    plot = plt.plot(f[:2000], Y_mag[:2000]) # magnitude spectrum (plot first 2000 frequencies)
    
    plt.xlabel('Frequency (Hz)') # labeling the axis
    plt.ylabel('Amplitude (Pa)')
    plt.title(title)
    
    plt.savefig(f'{title[:-4]}.jpg')  # save graph image
    #plt.show() # display the graph
    #print(title[:-3])

    #plt.close() # close the graph
    return Y_mag, f

def get_avg_graph(Y_mag, f_mag, title):
    ### SHORT FOURIER TRANSFORM ###
    # stft = np.abs(librosa.stft(y))
    # no librosa -> switch to scipy
    plt.figure(figsize=(14,5))
    plot = plt.plot(f[:2000], Y_mag[:2000]) # magnitude spectrum (plot first 2000 frequencies)
    
    plt.xlabel('Frequency (Hz)') # labeling the axis
    plt.ylabel('Amplitude (Pa)')
    plt.title(title)
    
    plt.savefig(f'{title[:-4]}.jpg')  # save graph image
    plot.show() # display the graph


flute_path = './audio_files/flute'
violin_path = './audio_files/violin'

# create a list of all the files in the directory
flute_list = os.listdir(flute_path)
violin_list = os.listdir(violin_path)

item_list = flute_list + violin_list
item_list = os.listdir(violin_path)
item_list.remove('.DS_Store')
print(f'done: {item_list}')

#item_list =['violin_d5(4).m4a', 'violin_d5(5).m4a']

# iterate through all the files and get the graphs

# Y_mag_avg = []
# f_avg = []

# Y_mag_avg_all = []
# f_avg_all = []
# counter = 0
for item in item_list:
    #print(str(item))

    y, sr = '', ''
    if "flute" in str(item):
        y, sr = load_audio(f'{flute_path}/{item}')   
    if "violin" in str(item):
        print("true")
        y, sr = load_audio(f'{violin_path}/{item}')

    print(f'y: {y}, sr: {sr}')
    Y_mag, f = get_graphs(y, sr, item)    

    # Y_mag_avg.append(Y_mag)
    # f_avg.append(f)


    # if counter % 4 == 0:
    #     Y_mag_temp = []
    #     f_temp = []

    #     for i in range(len(Y_mag_avg[0])):
    #         for j in range(len(Y_mag_avg)):
    #             Y_mag_temp.append(Y_mag_avg[j][i])
    #             f_temp.append(f_avg[j][i])

    #         Y_mag_avg_all.append([np.average(Y_mag_temp)])
    #         f_avg_all.append([np.average(f_temp)])

    #         Y_mag_temp = []
    #         f_temp = []

    #     Y_mag_avg_new = np.average(np.array(Y_mag_avg))
    #     f_avg_new = np.average(np.array(f_avg))

    #     Y_mag_avg_all.append(Y_mag_avg_new)
    #     f_mag_all.append(f_avg_new)

    #     Y_mag_avg = []
    #     f_avg = []
# print(f'Y_max_avg_all: {Y_mag_avg_all}')
# print(f'f_avg_all: {f_avg_all}')


### AMPLITUDE VS TIME GRAPH ###
# fig, ax = plt.subplots(nrows=1, sharex=True)
# librosa.display.waveshow(y, sr=sr)
# ax.set(title='Envelope view, mono')
# ax.label_outer()
# plt.title('graph 1')
# plt.show()

### SPECTRAL GRAPH ###
# spec = librosa.feature.melspectrogram(y=y, sr=sr)
# # librosa.display.specshow(spec,y_axis='mel', x_axis='s', sr=sr)
# db_spec = librosa.power_to_db(spec, ref=np.max,)
# librosa.display.specshow(db_spec, y_axis='mel', x_axis='s', sr=sr)
# plt.colorbar()
# plt.show()

# POWER SPECTROGRAM FREQ VS TIME W AMP IN COLORS #
# fig, ax = plt.subplots()
# img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), sr=sr, x_axis='time', y_axis='log')
# ax.set_title('Power spectrogram')
# fig.colorbar(img, ax=ax, format='%+2.0f dB')
# plt.show()

# FOURIER TRANSFORM AMP VS FREQ #
# amp_vs_freq_1000 = []
# for i in range(len(stft)):
#     amp_vs_freq_1000.append([stft[i][40]])
#print(amp_vs_freq_1000)
# fig, ax = plt.subplots()
# img = librosa.display.specshow()
