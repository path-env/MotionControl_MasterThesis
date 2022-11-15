import numpy as np
import time
import matplotlib.pyplot as plt

import cv2
import random
from sklearn.model_selection import train_test_split
# import tensorflow as tf
import torch
import torch.optim as optim
from data.params import OCIParams
from models.EEGNet_2018 import EEGnet
from carla_bci_control.src.carla_bci_control.src.oci_interface import OpenBciInterface

class SimEnv():
    def __init__(self, Expr_name = 'dummy') -> None:
        # Pytorch +BCI
        jit_model = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/EEGnet.pt'
        # jit_model = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/ATTNnet_md_script.pt'
        self.bci = OpenBciInterface(jit_model)
        self.bci._lsl_comm()
        self.dCfg = OCIParams()
        self.events = self.dCfg.event_dict_rec #['right', 'left', 'none']#,'up', 'down']
        FFT_MAX_HZ = 60
        # HM_SECONDS = 30 # this is approximate. Not 100%. do not depend on this.
        self.TOTAL_ITERS = 30 # ~25 iters/sec
        BOX_MOVE = "model"  # random or model
        # last_print = time.time()
        # fps_counter = deque(maxlen=150)
        self.env_init()
        self.EXPR_NAME = Expr_name 

    def env_init(self):
        self.WIDTH = 800
        self.HEIGHT = 800
        SQ_SIZE = 50
        self.MOVE_SPEED = 20
        self.car = {'x1': int(int(self.WIDTH)/2-int(SQ_SIZE/2)), 
                'x2': int(int(self.WIDTH)/2+int(SQ_SIZE/2)),
                'y1': int(int(self.HEIGHT)/2-int(SQ_SIZE/2)),
                'y2': int(int(self.HEIGHT)/2+int(SQ_SIZE/2))}

        self.square = {'x1': int(int(self.WIDTH)/2-int(SQ_SIZE/2)), 
                'x2': int(int(self.WIDTH)/2+int(SQ_SIZE/2)),
                'y1': int(int(self.HEIGHT)/2-int(SQ_SIZE/2)),
                'y2': int(int(self.HEIGHT)/2+int(SQ_SIZE/2))}

        self.box1 = np.ones((self.car['y2']-self.car['y1'], self.car['x2']-self.car['x1'], 3)) * np.array([0.5, 0.5, 0.0])
        self.box2 = np.ones((self.square['y2']-self.square['y1'], self.square['x2']-self.square['x1'], 3)) * np.array([0.5, 1, 1])
        self.horizontal_line = np.ones((self.HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
        self.vertical_line = np.ones((10, self.WIDTH, 3)) * np.array([0,0,0])

    def env_reinit(self, action =None):
        if action == 'left':
            self.square = {'x1': 375-250,  'x2': 425-250,   'y1': 375-100,  'y2': 425-100}
        elif action == 'right':
            self.square = {'x1': 375+250,  'x2': 425+250,   'y1': 375-100,  'y2': 425-100}
        else:
            self.square = {'x1': 375,  'x2': 425,   'y1': 375,  'y2': 425}
            # self.box1 = np.ones((self.car['y2']-self.car['y1'], self.car['x2']-self.car['x1'], 3)) * np.array([0.5, 0.5, 0.0])

            # self.box2 = self.box1
        try:
            env = np.zeros((self.WIDTH, self.HEIGHT, 3))
            env[:,self.HEIGHT//2-5:self.HEIGHT//2+5,:] = self.horizontal_line
            env[self.WIDTH//2-5:self.WIDTH//2+5,:,:] = self.vertical_line
            env[self.car['y1']:self.car['y2'], self.car['x1']:self.car['x2']] = self.box1
            env[self.square['y1']:self.square['y2'], self.square['x1']:self.square['x2']] = self.box2
        except Exception as e:
            pass
        cv2.imshow('', env)
        cv2.waitKey(1)
  
    def sim_data(self):
        filename = 'OBCI_3cls_EEGnet_onlinetrn.npz'
        data = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data/train/{filename}')
        train_x = data['arr_0']
        train_y = data['arr_1']
        return train_x


    def train_getdata(self, train = True):
        total = 0
        left, up = 0,0
        right, down = 0,0
        none = 0
        correct = 0 

        channel_datas = []
        y = []
        BOX_MOVE = 'model'
        if train:
            device = 'cuda'
            dtype = torch.float32
            model = EEGnet(n_classes = 3, n_chan = 16, n_T = self.bci.n_samp,
                sf = self.dCfg.sfreq, F1 = 24, F2 = 16, D =4, dt = 0.5)
            model = model.to(device=device, dtype= dtype)
            optimizer = optim.RAdam(model.parameters(), lr=0.001)
            loss_fn = torch.nn.CrossEntropyLoss() 
            train_loss = 0.0
            model.train()

        actions = np.repeat(list(self.events.keys()), repeats=self.TOTAL_ITERS)
        np.random.shuffle(actions)
        try:
            event_t = []
            start = time.time()
        # -8s---IDLE -----3s---Prepare ---0s ----Move(& feedback)----- 3s-----Feedback ----4s---
            for cnt, ACTION in enumerate(actions):  # how many iterations. Eventually this would be a while True
                action = self.events[ACTION]
                
                self.env_init()
                time.sleep(2)
                # while (time.time() - now) < 12:
                print(f'...............IDLE.............{cnt}|{3*self.TOTAL_ITERS}')
                
                time.sleep(3) # Idle
                print(f'Now perform action: {ACTION}.........')
                self.env_reinit(ACTION)
                # event_t.append(time.time() - start)
                # while (time.time() - now) < 8:
                #     pass #  Prepare
                time.sleep(3) 
                # fps_counter.append(time.time() - last_print)
                # last_print = time.time()
                # cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
                # # print(cur_raw_hz)
                
                
                # choice = 2#np.random.random_integers(0,4)
                # while (time.time() - now) < 11: # Move(& feedback)
                if BOX_MOVE == "random":
                    move = random.choice([-1,0,1])
                    self.car['x1'] = max(-3, self.car['x1']-self.MOVE_SPEED)
                    self.car['x2'] = max(-3, self.car['x2']-self.MOVE_SPEED)
                    eeg_TS = 0

                elif BOX_MOVE == "model":
                    time.sleep(4) # Move and feedback
                    t,eeg_TS = self.bci.get_data_lsl()
                    
                    if train:
                        # Online Training
                        label = torch.tensor([action]).to(device, dtype = torch.int64)
                        optimizer.zero_grad(set_to_none= True)
                        outputs = model(eeg_TS)#.flatten()
                        loss = loss_fn(outputs, label)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * eeg_TS.size(0)            
                        prob = torch.softmax(outputs, dim =1)
                    else:
                        # online testing
                        prob, _ = self.bci.getCmd()

                    choice = torch.argmax(prob)
                    # print(choice)
                    if choice == 1:
                        if ACTION == "left":
                            # print('left')
                            correct += 1
                        self.car['x1'] = (375-250)#, self.car['x1']-self.MOVE_SPEED)
                        self.car['x2'] = (425-250)#, self.car['x2']-self.MOVE_SPEED)
                        left += 1

                    elif choice == 0:
                        if ACTION == "right":
                            # print('right')
                            correct += 1
                        self.car['x1'] = (375+250)#, self.car['x1']+self.MOVE_SPEED)
                        self.car['x2'] = (425+250)#, self.car['x2']+self.MOVE_SPEED)
                        right += 1

                    # elif choice == 2:
                    #     if ACTION == "UP":Raw_
                    #     self.car['y1'] -= self.MOVE_SPEED
                    #     self.car['y2'] -= self.MOVE_SPEED
                    #     up += 1

                    # elif choice == 3:
                    #     if ACTION == "down":
                    #         print('down')
                    #         correct += 1
                    #     self.car['y1'] += self.MOVE_SPEED
                    #     self.car['y2'] += self.MOVE_SPEED
                    #     down += 1

                    else:
                        if ACTION == "none":
                            # print('none')
                            correct += 1
                        none += 1
                    self.env_reinit()
                    # time.sleep(1) # Feedback
                    # t,eeg_TS = self.bci.get_data_lsl()
                    event_t.append(t)
                # while (time.time() - now) < 12:
                #     pass
                total += 1
                channel_datas.append(eeg_TS.cpu().numpy())
                y.append(action)
                # time.sleep(1)

        finally:
            #plt.plot(channel_datas[0][0])
            #plt.show()
            model_scripted = torch.jit.script(model) 
            model_scripted.save(f"/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/{model._get_name()}Online.pt") 
            print(f"saving model...")

            datadir = "/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data"
            # Raw data
            raw_data = np.stack(channel_datas, axis =1)[0].squeeze(1)
            x = np.hstack(raw_data)
            event_t =  np.array(event_t).flatten() - event_t[0][0]
            idx = [(8*self.bci.sfreq*i) for i in range(raw_data.shape[0])]
            event_t = event_t[idx]
            np.savez(f'{datadir}/Raw_{self.EXPR_NAME}',x, event_t, y)
            print(f"saving Raw data...")

            # Epoched data
            channel_datas = np.stack(channel_datas, axis =1)[0]
            # channel_datas = np.expand_dims(channel_datas, axis=1)
            # Test train split
            train_x,test_x,train_y,test_y = train_test_split(channel_datas, y, test_size= self.dCfg.test_split,
                                            stratify= None, random_state= None, shuffle=False)
            print(channel_datas.shape)
            # labels = labels[ACTION]*np.ones(self.TOTAL_ITERS)
            np.savez(f'{datadir}/train/{self.EXPR_NAME}',train_x, train_y)
            np.savez(f'{datadir}/test/{self.EXPR_NAME}',test_x, test_y)
            # np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
            print(f"saving Epoched data...")

            print(ACTION, correct/total)
            print(f"left: {left/total}, right: {right/total}, up: {up/total}, down: {down/total}, none: {none/total}")

if __name__ =='__main__':
    env = SimEnv()
    env.train_getdata()