from mne_realtime import LSLClient, MockLSLStream
import mne
import matplotlib.pyplot as plt

from data.params import  OCIParams
import  data.brain_atlas as bm

if __name__ == '__main__':
    wait_max = 5
    dCfg = OCIParams()
    sfreq = dCfg.sfreq
    ch_names = bm.oci_Channels
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types);

    _, ax = plt.subplots(1)
    n_epochs = 5
    host = 'openbcigui'

    client = LSLClient(info=None, host=host, wait_max=wait_max)
    client.start()
    r = client.get_data_as_epoch(1000)
    print(r)   
    client.stop()

#     with LSLClient(info=None, host=host, wait_max=wait_max) as client:
#         client_info = client.get_measurement_info()
#         # client_info['sfreq'] = ch_names
#         sfreq = int(client_info['sfreq'])
#         print(client_info)
#         r = client.get_data_as_epoch(1000)
#         print(r)
#         # # let's observe ten seconds of data
#         # for ii in range(n_epochs):
#         #     print('Got epoch %d/%d' % (ii + 1, n_epochs))
#         #     plt.cla()
#         #     epoch = client.get_data_as_epoch(n_samples=sfreq)
#         #     epoch.average().plot(axes=ax)
#         #     plt.pause(1.)
#         # plt.draw()
# print('Streams closed')