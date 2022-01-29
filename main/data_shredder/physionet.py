import numpy as np
import mne

folderpath = "/media/mangaldeep/HDD3/DataSets/PhysioNet/S001/"
filename = '1.npy'

def PhysioNet(folderPath):
    gt = []
    full_data = np.empty((1,64))
    clss = np.array(['T0', 'T1', 'T2'])
    for fileno in range(1,15):
        file = folderPath+ "S001R" + str(fileno).zfill(2) +".edf"
        data = mne.io.read_raw_edf(file)
        raw_data = data.get_data()
        print(raw_data.shape)
        full_data = np.vstack((full_data,raw_data.T))
        # info = data.info
        # channels = data.ch_names
        i = 0
        for t in data.times:
            if i<len(data.annotations.onset) and data.annotations.onset[i] == t:
                tag = data.annotations.description[i]
                i+=1
            gt.append(np.where(tag==clss)[0][0])
        full_data = full_data[1:,:]
    return full_data, gt

def extractPhysioNet(time_window, moving, n_classes = 10, no_channels = 64):
    input = np.load(folderpath+filename)
    restlabel = 10
    id = np.where(input[:,-1]!=10)[0]
    input = input[id,:]
    xx = input[:, :no_channels]
    yy = input[:, no_channels:no_channels + 1]
    new_x = []
    new_y = []
    number = int((xx.shape[0] / moving) - 1)
    for i in range(number):
        ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])
        if ave_y in range(n_classes + 1):
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(ave_y)
        else:
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(0)

    new_x = np.array(new_x)
    # new_x = new_x.reshape([-1, no_channels * time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    # data = np.hstack((new_x, new_y))
    ip = np.vstack((new_x, new_x[-1]))  # add the last sample again, to make the sample number round
    label = np.vstack(new_y, new_y[-1])
    return ip, label