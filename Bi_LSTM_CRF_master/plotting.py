# -*- coding:utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

###############plotting the result of sentence-based model##############################
results_train=pd.read_csv('./intermediate/train_sent.csv', header=None, delimiter=',')
results_train=results_train.values

print(np.mean(results_train,axis=0))


with open('./intermediate/valid_res_sent.pkl', 'rb') as f:
    results_valid = pickle.load(f, encoding='latin1')
    f.close()

print(np.mean(results_valid[:176],axis=0))


with open('./intermediate/test_res_sent.pkl', 'rb') as f:
    results_test = pickle.load(f, encoding='latin1')
    f.close()

print(np.mean(results_test[:176],axis=0))


D=range(1,177)
clsf=['precision','recall','f1']
# Plot the result of experiment 4.
fig, ax = plt.subplots(3)


ax[0].plot(D,results_train[:176*9:9,0],'.-',label=clsf[0])
ax[0].plot(D,results_train[:176*9:9,1],'.-',label=clsf[1])
ax[0].plot(D,results_train[:176*9:9,2],'.-',label=clsf[2])
ax[0].legend( loc=1, fontsize='small')

ax[1].plot(D,results_valid[:176,0],'.:',label=clsf[0])
ax[1].plot(D,results_valid[:176,1],'.:',label=clsf[1])
ax[1].plot(D,results_valid[:176,2],'.:',label=clsf[2])
ax[1].legend( loc=1, fontsize='small')

ax[2].plot(D,results_test[:176,0],'.-',label=clsf[0])
ax[2].plot(D,results_test[:176,1],'.-',label=clsf[1])
ax[2].plot(D,results_test[:176,2],'.-',label=clsf[2])
ax[2].legend( loc=1, fontsize='small')

ax[0].set_ylabel('Training set')
ax[1].set_ylabel('Validation set')
ax[2].set_ylabel('Test set')

ax[2].set_xlabel('Index of data batch(1000 sentences/batch) ')

ax[0].axis([0.5, 175, 0., 0.5])
ax[1].axis([0.5, 175, 0., 0.5])
ax[2].axis([0.5, 175, 0., 0.5])


plt.show()

###################plotting the result of sentence-based model########################

results_train=pd.read_csv('./intermediate/train_doc.csv', header=None, delimiter=',')
results_train=results_train.values
print(np.mean(results_train,axis=0))

with open('./intermediate/valid_res.pkl', 'rb') as f:
    results_valid = pickle.load(f, encoding='latin1')
    f.close()


print(np.mean(results_valid[:152],axis=0))


with open('./intermediate/test_res.pkl', 'rb') as f:
    results_test = pickle.load(f, encoding='latin1')
    f.close()

print(np.mean(results_test[:150],axis=0))
D=range(1,153)
clsf=['precision-','recall','f1']

fig, ax = plt.subplots(3)
# Plot the result of experiment 4.
ax[0].plot(D,results_train[:152*8:8,0],'.-',label=clsf[0])
ax[0].plot(D,results_train[:152*8:8,1],'.-',label=clsf[1])
ax[0].plot(D,results_train[:152*8:8,2],'.-',label=clsf[2])
ax[0].legend( loc=1, fontsize='small')

ax[1].plot(D,results_valid[:152,0],'.:',label=clsf[0])
ax[1].plot(D,results_valid[:152,1],'.:',label=clsf[1])
ax[1].plot(D,results_valid[:152,2],'.:',label=clsf[2])
ax[1].legend( loc=1, fontsize='small')

ax[2].plot(D,results_test[:152,0],'.-',label=clsf[0])
ax[2].plot(D,results_test[:152,1],'.-',label=clsf[1])
ax[2].plot(D,results_test[:152,2],'.-',label=clsf[2])
ax[2].legend( loc=1, fontsize='small')

ax[0].set_ylabel('Training set')
ax[1].set_ylabel('Validation set')
ax[2].set_ylabel('Test set')

ax[2].set_xlabel('Index of data batch(100 documents/batch)')

ax[0].axis([0.5, 150, 0., 0.8])
ax[1].axis([0.5, 150, 0., 0.8])
ax[2].axis([0.5, 150, 0., 0.8])

plt.show()
print()