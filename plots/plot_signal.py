
import numpy as np
import mne
import matplotlib.pyplot as plt

channel = 4
sampling_rate = 128

ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
ch_types = ['eeg'] * 14
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_rate)

# data = np.load('/home/airlab/Desktop/EEG/dataset/amigos/amigos_1s_norm/S14T06_0015.npy')

# Init noise
np.random.seed(0)
data = np.random.uniform(low=0, high=1, size=(channel, sampling_rate))

uap_file_name = '/home/airlab/Desktop/EEG/code/eeg_uap_airlab/result/amigos_eegnet/uap/0.0392/uap_air_exam_nt_fold0.npy'
perturbation = np.load(uap_file_name)
perturbation = np.reshape(perturbation, (14, 128))

#data = data + perturbation
data = perturbation
print(data.shape)
data1 = data[:1]
data2 = data[1:2]
data3 = data[2:3]
data4 = data[3:4]
data5 = data[4:5]
data6 = data[5:6]
data7 = data[6:7]
data8 = data[7:8]
data9 = data[8:9]
data10 = data[9:10]
data11 = data[10:11]
data12 = data[11:12]
data13 = data[12:13]
data14 = data[13:14]

data1 = np.reshape(data1, (sampling_rate, -1))
data2 = np.reshape(data2, (sampling_rate, -1))
data3 = np.reshape(data3, (sampling_rate, -1))
data4 = np.reshape(data4, (sampling_rate, -1))
data5 = np.reshape(data5, (sampling_rate, -1))
data6 = np.reshape(data6, (sampling_rate, -1))
data7 = np.reshape(data7, (sampling_rate, -1))
data8 = np.reshape(data8, (sampling_rate, -1))
data9 = np.reshape(data9, (sampling_rate, -1))
data10 = np.reshape(data10, (sampling_rate, -1))
data11 = np.reshape(data11, (sampling_rate, -1))
data12 = np.reshape(data12, (sampling_rate, -1))
data13 = np.reshape(data13, (sampling_rate, -1))
data14 = np.reshape(data14, (sampling_rate, -1))

data_list = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14]

fig = plt.figure()
gs = fig.add_gridspec(channel, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
#fig.suptitle('Benign Examples')
#fig.suptitle('Adversarial Examples')
fig.suptitle('Adversarial perturbations')
#fig.suptitle('Init noise')

color_list = ['red', 'orange', 'gold', 'green', 'yellow', 'black', 'pink', 'blue', 'olive', 'brown', 'indigo', 'gray', 'tan', 'darkgreen']
colors = plt.rcParams["axes.prop_cycle"]()

for i in range(channel):
    c = next(colors)["color"]
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].set(ylabel = ch_names[i])
    axs[i].plot(data_list[i], color=c)


# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()
    ax.set_yticks([])
    ax.set_xticks([])

plt.show()
plt.draw()
#fig.savefig('AMIGOS_benign.png')
#fig.savefig('AMIGOS_adversarial.png')
fig.savefig('AMIGOS_perturbation.png')
#fig.savefig('AMIGOS_init_noise.png')

