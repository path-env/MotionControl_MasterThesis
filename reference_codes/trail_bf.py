import time
from tkinter.tix import Tree
import numpy as np
import matplotlib
# matplotlib.use ('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import brainflow as bf
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# from mne.viz.topomap import _prepare_topo_plot, plot_topomap
import mne
# from mne.channels import read_layout

# from mpl_toolkits.axes_grid1 import make_axes_locatable



# BoardShim.enable_dev_board_logger ()
# use synthetic board for demo
params = BrainFlowInputParams ()

params.timeout = 10
params.serial_port = '/dev/ttyUSB0'


board = BoardShim (BoardIds.CYTON_DAISY_BOARD, params)
board.prepare_session()
board.start_stream()
time.sleep (1)
# print(board.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD))
# while True:
data = board.get_current_board_data(num_samples=200)
print(data.shape)
time.sleep (1)


# except KeyboardInterrupt:
#     board.stop_stream ()
#     board.release_session ()

eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
print(eeg_channels)
eeg_data = data[eeg_channels, :]
# plt.plot(eeg_data)
# plt.show()
for i in range(len(eeg_channels)):
    bf.DataFilter.perform_bandpass(eeg_data[i], 125 , 0.5, 50,4, 0, 0.1)
# plt.figure()
# plt.plot(eeg_data)
# plt.show()
eeg_data = eeg_data / 1000000 # BrainFlow returns uV, convert to V for MNE
print(eeg_data.shape)


# Creating MNE objects from brainflow data arrays
# ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
# ch_names = ['T7', 'CP5', 'FC5', 'C3', 'C4', 'FC6', 'CP6', 'T8']

ch_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'O1', 'O2', 'P4', 'T7', 'T8', 'F7', 'F8', 'P7', 'P8', 'Fp1','Fp2']
ch_type = []
ch_types = [ch_type.append('eeg') for _ in range(len(ch_names))]
sfreq = BoardShim.get_sampling_rate (BoardIds.SYNTHETIC_BOARD.value)
info = mne.create_info (ch_names = ch_names, sfreq = sfreq, ch_types = ch_type)
raw = mne.io.RawArray (eeg_data, info)
# its time to plot something!
raw.plot();
raw.plot_psd (average = True);
# plt.savefig ('psd.png')
