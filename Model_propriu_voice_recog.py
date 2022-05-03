#!/usr/bin/env python
# coding: utf-8

# ## Voice recognition system

# In[1]:


import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# In[2]:


data_path = ".\\data\\"
meta_file = "rec.txt"
sig_len = 88200 // 2
batch_size = 24


# In[3]:


def normalization(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def standardization(arr):
    return (arr - np.mean(arr)) / np.std(arr)

def get_fft(arr):
    data_ftt = np.fft.fft(arr)
    data_ftt = np.abs(data_ftt.real)[:len(arr)//2]
    return data_ftt

# de gandit cev ce nu crapa
def augment_data(X):
    for i in  range(len(X)):
        rv = np.random.rand(1)
        X[i] = X[i] * rv[0] * np.random.choice([0.5, 1.0, 1.1])
    return X

def fft2img(data):
    _, _, _, ret = plt.specgram(x=data.flatten(), NFFT=256, Fs=22050)
    return ret.make_image(None)[0][:,:,:3]

# Load all the data into a meta object
f = open(data_path + meta_file, mode="r")
lines = f.readlines()
f.close()

meta_obj = []
for line in lines:
    dobj = json.loads(line)
    meta_obj.append(dobj)
    
to_aug = np.random.choice(meta_obj,size=300)
to_aug_obj = []
i = 600
for rec in to_aug:
    label = rec['label']
    sound = np.load(data_path + rec['file'])
    sound[len(sound) // 2:] = augment_data(sound[len(sound) // 2:])
    fn = "sound_" + str(i)
    ts = int(datetime.now().timestamp())
    np.save(data_path + str(fn),sound)
    str_to_save = '{' + '"id":"' + str(ts) + '","file":"' + str(fn) + '.npy","label":"' + str(label) + '"}'
    dobj = json.loads(str_to_save)
    print(dobj)
    to_aug_obj.append(dobj)
    i += 1

    
meta_obj.extend(to_aug_obj)
#meta_obj.extend(to_aug)
meta_obj = np.array(meta_obj)
#np.append(meta_obj,to_aug)
np.random.shuffle(meta_obj)
print(meta_obj)

# create a helper class to load a batch of data on request
# idx = 10, pos = 10 * bs
class DataLoader(keras.utils.Sequence):
    def __init__(self, data, batch_size, data_path, shuffle=True):
        self.data = np.array(data)
        self.batch_size = batch_size
        self.data_path = data_path
        self.shuffle = shuffle
        if self.shuffle == True:
            np.random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)//self.batch_size
    
    # get a signle batch in memory
    def __getitem__(self, idx):
        data_batch = self.data[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = []
        y = []
        for db in data_batch:
            snd_data = np.load(self.data_path + db['file'])
            snd_label = self.__to_categorical(db['label'])
            
            # preprocessing
            # aug_cond = np.random.choice([0,1])
            # if aug_cond == 1:
            #     snd_data = augment(snd_data)
            snd_data = normalization(snd_data)
            snd_data = get_fft(snd_data)
            snd_data = fft2img(snd_data)
            #normalizare dupa fft?
            
            X.append(snd_data)
            y.append(snd_label)
            
        X = np.array(X)
        y = np.array(y)
        
        return X,y
    
    def __to_categorical(self, data):
        ret_val = 0
        if data == "open":
            ret_val = np.array([1,0,0])
        elif data == "close":
            ret_val = np.array([0,1,0])
        else:
            ret_val = np.array([0,0,1])
        return ret_val
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.data)
    
    
# to_aug = np.random.choice(meta_obj, size=300)
# meta_obj.extend(to_aug)
# meta_obj = np.array(meta_obj)
np.random.shuffle(meta_obj)

training_vals, validation_vals, test_vals = np.split(meta_obj, [int(len(meta_obj)*0.75), int(len(meta_obj)*0.95)])

training_data = DataLoader(training_vals, batch_size, data_path)
validation_data = DataLoader(validation_vals, batch_size, data_path)
test_data = DataLoader(test_vals, batch_size, data_path)

print("Data len:" ,len(meta_obj), " Train data len:", len(training_data), " Val data len:", len(validation_data), " Test data:", len(test_vals))

sample = training_data.__getitem__(5)
plt.imshow(sample[0][0])

print(f"Sample - len: {len(sample)}; shape[0]: {sample[0].shape}; shape[1]: {sample[1].shape}; shape[0][0]: {sample[0][0].shape}")
print(f"Sample[0] - {sample[0]}; Sample[1]: {sample[1]}")
#half_sample = sample[0][0][:sample[0][0].shape[0] // 2]

#print(f"half_sample - len: {len(half_sample)}; shape[0]: {half_sample[0].shape}; shape[1]: {half_sample[1].shape}; shape[0][0]: {half_sample[0][0].shape}")

# arr = []
# for i in range (len(training_data)):
#     sample = test_data.__getitem__(i)
#     for j in range (len(sample[0])):
#         y = sample[1][j]
#         ytt = np.argmax(y)
#         arr.append(ytt)
        
# plt.hist(arr)


# In[4]:


# input layer takes the signal lenght (not the sample number)

IM_H = 335
IM_W = 218
IM_CH = 3
data_in = keras.Input(shape=(IM_W, IM_H, IM_CH))

val = keras.layers.Conv2D(16, 17, activation=keras.activations.relu)(data_in)
val = keras.layers.BatchNormalization()(val)
val = keras.activations.relu(val)

val = keras.layers.Dropout(0.5)(val)
val = keras.layers.MaxPool2D((2,2))(val)

val = keras.layers.Conv2D(8, (5,5), activation=keras.activations.relu)(val)
val = keras.layers.BatchNormalization()(val)
val = keras.activations.relu(val)

# mai merge un conv2D pe aici pe undeva

val = keras.layers.Dropout(0.5)(val)
val = keras.layers.MaxPool2D((2,2))(val)

# val = keras.layers.Conv2D(3, (3,3))(val)
# val = keras.layers.BatchNormalization()(val)
# val = keras.activations.relu(val)


#-------------------------------------------------------------------------------------------------------
# data_in = keras.Input(shape=(sig_len,1))

# val = keras.layers.Dense(5,)(data_in)
# val = keras.layers.BatchNormalization()(val)
# val = keras.activations.relu(val)

# val = keras.layers.Dropout(0.5)(val)

# val = keras.layers.Conv1D(32, 3, activation=keras.activations.relu, name="conv1d_0", strides=2)(val)
# # val = keras.layers.Dense(30,)(val)
# val = keras.layers.BatchNormalization()(val)
# val = keras.activations.relu(val)

# #val = keras.layers.MaxPool1D(pool_size=2)(val)

# # val = keras.layers.Dropout(0.5)(val)

# val = keras.layers.Conv1D(32, 3, activation=keras.activations.relu, name="conv1d_1")(val)

# # val = keras.layers.Dense(10,)(val)
# val = keras.layers.BatchNormalization()(val)
# val = keras.activations.relu(val)

val = keras.layers.Dense(30,)(val)
val = keras.layers.BatchNormalization()(val)
val = keras.activations.relu(val)
#--------------------------------------------------------------------------------------------------------
val = keras.layers.Flatten()(val)

out = keras.layers.Dense(3, activation=keras.activations.softmax)(val)

#construction of the model(inputs, outputs)
model = keras.Model(inputs=data_in, outputs=out)
model.summary()

keras.utils.plot_model(model, to_file='.\\st_nn.png', show_shapes=True, rankdir="TD")

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['acc'])

print(data_in.shape, out.shape)


# In[5]:


# Let's train the model
epochs = 100

cb = [EarlyStopping(patience=3, min_delta=0.0000000001), ModelCheckpoint(filepath="vrs.h5", save_best_only=True, verbose=1)]

hist = model.fit(x=training_data, validation_data=validation_data, shuffle=True, epochs=epochs, callbacks=cb)


# # Home work #4
# - construct the model and train on the data, then test it on test_data
# - provide the accuracy value (maximal one), f1_score

# In[14]:


from sklearn.metrics import classification_report

print(test_data.__getitem__(0)[0].shape, test_data.__getitem__(0)[1].shape, len(test_data)) 
print(test_data)

yt = []
yp = []
for i in range (len(test_vals)):
    sample = test_data.__getitem__(i)
    for j in range (len(sample[0])):
        X = sample[0][j].reshape(1,IM_W, IM_H, IM_CH)
        y = sample[1][j]
        ret = model.predict(X)
        ytt = np.argmax(y)
        ypt = np.argmax(ret)
        yt.append(ytt)
        yp.append(ypt)

print(classification_report(yt, yp, labels=[0,1,2]))


# In[21]:


fig, axs = plt.subplots(2, 1)

accu = np.max(hist.history['acc'])
vaccu = np.max(hist.history['val_acc'])
print('accu: ', accu, 'val_accu: ', vaccu)

axs[0].plot(hist.history['acc'])
axs[0].plot(hist.history['val_acc'])


axs[1].plot(hist.history['loss'])
axs[1].plot(hist.history['val_loss'])


# In[22]:


print(yp)


# In[23]:


dict_response = {
    0: 'open',
    1: 'close',
    2: 'noise'
}

fs = 44100
duration = 2

to_predict = input('Do you want to predict? [y/n]')
while(to_predict == 'y'):
    print('Recording...')
    data = sd.rec(duration * fs, samplerate=fs, channels=1)
    sd.wait()
    snd_data = normalization(data)
    snd_data = get_fft(snd_data)        
    snd_data = fft2img(snd_data)
    snd_data = snd_data.reshape(1,IM_W, IM_H, IM_CH)
    ret = model.predict(snd_data)
    print(ret.flatten().tolist())
    rez = 2
    ret = np.round(ret.flatten()).tolist()
    #print(ret)
    for i in range(len(ret)):
        if ret[i] == 1:
            rez = i
            break
    print(f'You said: {dict_response[rez]}')
    to_predict = input('Do you want to predict? [y/n]')


# In[ ]:




