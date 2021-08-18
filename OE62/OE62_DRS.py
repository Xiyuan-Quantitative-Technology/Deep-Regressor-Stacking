# -*- coding: utf-8 -*-
"""

@author: tanzheng
"""
import torch
import numpy as np
import pickle
import time
from two_layer_method import Two_layer_meth
import meta_X_transform as mXt
import matplotlib.pyplot as plt
import pandas as pd
##########################################################
# epoch parameter
epoch_No = [350,320,70,330,360,260,340,380,300,60]
# epoch_No = [1]
##########################################################
# epoch_No = [100, 100]
# set seed and device
seed = 12
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Python random module.
torch.manual_seed(seed)

device = torch.device("cpu")
torch.set_num_threads(40)

##########################################################

# load dataset
with open('OE62_2048ECFP_stand_input.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()
    
tasks, train_dataset, val_dataset, test_dataset = data_list

# training
train_y = train_dataset[:,2048:2058]

test_y = test_dataset[:,2048:2058]

val_y = val_dataset[:,2048:2058]
# y_train_df = pd.DataFrame(train_y)
# y_train_df.to_csv('y_train.csv', header=0, index=None)

# y_test_df = pd.DataFrame(test_y)
# y_test_df.to_csv('y_test.csv', header=0, index=None)
#training
X_train = train_dataset[:,0:2048]
train_X = torch.Tensor(X_train).to(device)

X_test = test_dataset[:,0:2048]
test_X  = torch.Tensor(X_test).to(device)

y_train_dataset = train_y
y_train_dataset = torch.Tensor(y_train_dataset).to(device)

y_val_dataset = val_y
y_val_dataset = torch.Tensor(y_val_dataset).to(device)

##########################################################
LR = 0.001
# first stage training
X = train_dataset[:,0:2048]
prop_y = train_y

X = torch.Tensor(X).to(device)
prop_y = torch.Tensor(prop_y).to(device)


y_val_dataset = val_y
y_val_dataset = torch.Tensor(y_val_dataset).to(device)

# start_time = time.time()
# First_stage_models = []
# for i in range(len(tasks)):
#       train_y_target = prop_y[:, i].reshape(y_train_dataset.shape[0],1)
#       # training data prepare
#       neural_nets = Two_layer_meth()
#       neural_nets.set_epoch(epoch_No[i])
#       neural_nets.set_Net(2048, 2048, 1024, 1)

#       model = neural_nets.two_layer_net(train_X,train_y_target)
#       First_stage_models.append(model)
# # end_time = time.time()
# # print(end_time - start_time)
#   # meta-variable generation
# y_meta_var = []
#   # loss_func = torch.nn.MSELoss()
# for i in range(len(tasks)):
#       y_train_pred = First_stage_models[i](X)
#       y_train_pred = y_train_pred.detach().numpy()
#       y_meta_var.append(y_train_pred)

# all_y_train_pred = np.empty(shape=(len(y_train_pred), 0))

# for i in range(len(tasks)):
#       all_y_train_pred = np.hstack((all_y_train_pred, y_meta_var[i]))

# y_train_df = pd.DataFrame(all_y_train_pred)
# y_train_df.to_csv('1_all_y_train_pred.csv', header=0, index=None)

# # # first stage out-sample prediction
# out_X = test_X
out_prop_y = test_y

# out_X = torch.Tensor(out_X).to(device)
# out_prop_y = torch.Tensor(out_prop_y).to(device)

# first_pred_out_y = []
# for i in range(len(tasks)):
#       y_test_pred = First_stage_models[i](out_X)
#       y_test_pred = y_test_pred.detach().numpy()

#       first_pred_out_y.append(y_test_pred)

# all_y_test_pred = np.empty(shape=(len(y_test_pred), 0))

# for i in range(len(tasks)):
#       all_y_test_pred = np.hstack((all_y_test_pred, first_pred_out_y[i]))

# y_test_df = pd.DataFrame(all_y_test_pred)
# y_test_df.to_csv('1_all_y_test_pred.csv', header=0, index=None)
# end_time = time.time()
# print(end_time - start_time)
##################################################################
# second stage training
deep_rrmse = []
deep_pred_all = []
deep_epoch = 10
for q in range(1, deep_epoch+1):
    meta_data = np.empty(shape=(len(X_train), 0))
    for qq in range(q):
        meta_data = np.hstack((meta_data,np.array(pd.read_csv(str(qq+1) + '_all_y_train_pred.csv', header=None))))
        # meta-feature generation
    meta_X = np.hstack((X_train, meta_data))
    meta_X = torch.Tensor(meta_X).to(device)
    l = meta_X.shape[1]
    
    start_time = time.time()
    Second_stage_models = []
    for i in range(len(tasks)):
        test_y_target = train_y[:, i]
        # training data prepare
        prop_y_train = prop_y[:,i].reshape(prop_y.shape[0],1)

        neural_nets = Two_layer_meth()
        neural_nets.set_epoch(epoch_No[i])
        neural_nets.set_Net(l, l, l/2, 1)

        model = neural_nets.two_layer_net(meta_X,prop_y_train)
        Second_stage_models.append(model)
    # end_time = time.time()
    # print(end_time - start_time)

    y_meta_var = []
    # loss_func = torch.nn.MSELoss()
    for i in range(len(tasks)):
        y_train_pred = Second_stage_models[i](meta_X)
        y_train_pred = y_train_pred.detach().numpy()
        y_meta_var.append(y_train_pred)

    all_y_train_pred = np.empty(shape=(len(y_train_pred), 0))

    for i in range(len(tasks)):
        all_y_train_pred = np.hstack((all_y_train_pred, y_meta_var[i]))

    y_train_df = pd.DataFrame(all_y_train_pred)
    y_train_df.to_csv(str(q+1) + '_all_y_train_pred.csv', header=0, index=None)

    # second stage out-sample prediction
    # meta-feature generation
    
    test_meta_data = np.empty(shape=(len(test_X), 0))
    for qq in range(q):
        test_meta_data = np.hstack((test_meta_data,np.array(pd.read_csv(str(qq+1) + '_all_y_test_pred.csv', header=None))))
        # meta-feature generation
    meta_out_X = np.hstack((test_X, test_meta_data))
    meta_out_X = torch.Tensor(meta_out_X).to(device)

    second_pred_out_y = []
    for i in range(len(tasks)):
        sec_y_test_pred = Second_stage_models[i](meta_out_X)
        sec_y_test_pred = sec_y_test_pred.detach().numpy()

        second_pred_out_y.append(sec_y_test_pred)

    all_y_second_pred = np.empty(shape=(len(sec_y_test_pred), 0))

    for i in range(len(tasks)):
        all_y_second_pred = np.hstack((all_y_second_pred, second_pred_out_y[i]))

    deep_pred_all.append(all_y_second_pred)

    y_second_df = pd.DataFrame(all_y_second_pred)
    y_second_df.to_csv(str(q+1) + '_all_y_test_pred.csv', header=0, index=None)

    ###############
    No_samples = out_prop_y.shape[0]
    # np_fir_pred_out_y = np.empty(shape=(No_samples, 0))
    np_sec_pred_out_y = np.empty(shape=(No_samples, 0))

    for i in range(len(tasks)):
        # np_fir_pred_out_y = np.hstack((np_fir_pred_out_y, first_pred_out_y[i]))
        np_sec_pred_out_y = np.hstack((np_sec_pred_out_y, second_pred_out_y[i]))

    ###########
    multi_task_RRMSE = []

    for i in range(len(tasks)):
        temp_MT_RRMSE = sum(np.square(out_prop_y[:, i]-np_sec_pred_out_y[:, i])) / sum(
            np.square(out_prop_y[:, i]-np.mean(out_prop_y[:, i])))
        temp_MT_RRMSE = np.sqrt(temp_MT_RRMSE)
        multi_task_RRMSE.append(temp_MT_RRMSE)

    deep_rrmse.append(multi_task_RRMSE)
    end_time = time.time()
    print(end_time - start_time)
##########################################################
# RRMSE PLOT DATA
##########################################################
out_prop_y = test_y
MT_predict_result = [out_prop_y, deep_rrmse, deep_pred_all, tasks]

with open('DRS_predict_seed12.pkl', 'wb') as f:
    pickle.dump(MT_predict_result, f)
    f.close()


