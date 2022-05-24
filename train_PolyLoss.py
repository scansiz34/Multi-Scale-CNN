import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from fusion_models.early_fusion import *
from data import *
import random
from sklearn.metrics import classification_report
from PolyLoss import to_one_hot, PolyLoss
import copy

torch.manual_seed(2021)  # cpu
torch.cuda.manual_seed(2021)  # gpu
np.random.seed(2021)  # numpy
random.seed(2021)  # random and transforms
torch.backends.cudnn.deterministic = True  

import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
cuda = torch.device('cuda:0')


def worker_init_fn(worker_id):
    np.random.seed(2021 + worker_id)


batch_size = 16
num_epochs = 20
target_names = ['class 0', 'class 1']
samples_train, labels_train, samples_test, labels_test, num_train_instances, num_test_instances = data_load()

msresnet = MSResNet(input_channel=1, layers=[1, 1, 1], num_classes=2)
msresnet = msresnet.cuda()
# criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
criterion = PolyLoss(softmax=False).cuda()

optimizer = torch.optim.Adam(msresnet.parameters(), lr=0.0001, weight_decay=0.000002)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
train_loss = np.zeros([num_epochs, 1])
test_loss = np.zeros([num_epochs, 1])
train_acc = np.zeros([num_epochs, 1])
test_acc = np.zeros([num_epochs, 1])


# data for training
dataset_train = Data.TensorDataset(samples_train, labels_train)
# identify data loader
dataset_train_loader = Data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                       pin_memory=True, worker_init_fn=worker_init_fn)


# data for testing
dataset_test = Data.TensorDataset(samples_test, labels_test)
# identify data loader
dataset_test_loader = Data.DataLoader(dataset=dataset_test, batch_size=60, shuffle=True, num_workers=0,
                                      pin_memory=True, worker_init_fn=worker_init_fn)

if __name__ == '__main__':
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        msresnet.train()
        scheduler.step()

        loss_x = 0
        correct_train = 0
        for i, (samples, labels) in enumerate(dataset_train_loader):
                samplesV = Variable(samples.cuda())
                samplesX0 = samplesV[:, 0:1, :, :]
#                 samplesX1 = samplesV[:, 1:2, :, :]
#                 samplesX2 = samplesV[:, 2:3, :, :]
#                 samplesX3 = samplesV[:, 3:4, :, :]
#                 samplesX4 = samplesV[:, 4:5, :, :]
                labels = labels.squeeze()
                labelsV = Variable(labels.cuda())
                optimizer.zero_grad()
#                 predict_label = msresnet(samplesX0, samplesX1, samplesX2, samplesX3, samplesX4)
                predict_label = msresnet(samplesX0)
#                 prediction = predict_label[0].data.max(1)[1]
                prediction = predict_label.data.max(1)[1]
                correct_train += prediction.eq(labelsV.data.long()).sum()

#                 loss = criterion(predict_label[0], labelsV)
                loss = criterion(predict_label, labelsV)
                loss_x += loss.item()
                loss.backward()
                optimizer.step()

        print("Training accuracy:", (100*float(correct_train)/num_train_instances))

        train_loss[epoch] = loss_x / num_train_instances
        train_acc[epoch] = 100*float(correct_train)/num_train_instances

        trainacc = str(100*float(correct_train)/num_train_instances)[0:6]

        loss_x = 0
        correct_test = 0
        y_true = []
        y_pre = []
        temp_true = []
        temp_pre = []

        msresnet.eval()
        for i, (samples, labels) in enumerate(dataset_test_loader):
            with torch.no_grad():
                samplesV = Variable(samples.cuda())
                samplesX0 = samplesV[:, 0:1, :, :]
#                 samplesX1 = samplesV[:, 1:2, :, :]
#                 samplesX2 = samplesV[:, 2:3, :, :]
#                 samplesX3 = samplesV[:, 3:4, :, :]
#                 samplesX4 = samplesV[:, 4:5, :, :]
                labels = labels.squeeze()
                labelsV = Variable(labels.cuda())
#                 predict_label = msresnet(samplesX0, samplesX1, samplesX2, samplesX3, samplesX4)
                predict_label = msresnet(samplesX0)
#                 prediction = predict_label[0].data.max(1)[1]
                prediction = predict_label.data.max(1)[1]
                correct_test += prediction.eq(labelsV.data.long()).sum()
                y_true.extend(labels.cpu().numpy())
                y_pre.extend(prediction.cpu().numpy())
#                 loss = criterion(predict_label[0], labelsV)
                loss = criterion(predict_label, labelsV)
                loss_x += loss.item()

        print("Test accuracy:", (100 * float(correct_test) / num_test_instances))

        test_loss[epoch] = loss_x / num_test_instances
        test_acc[epoch] = 100 * float(correct_test) / num_test_instances

        testacc = str(100 * float(correct_test) / num_test_instances)[0:6]

        if epoch == 0:
            temp_test = correct_test
            temp_train = correct_train
            temp_true = y_true
            temp_pre = y_pre
            print(classification_report(y_true, y_pre, digits=4, target_names=target_names), '\n')
            best_model_wts = copy.deepcopy(msresnet.state_dict())
            torch.save(best_model_wts, './models/multi_scale_cnn')
                
        elif correct_test >= temp_test:
            temp_test = correct_test
            temp_train = correct_train
            temp_epoch = 0
            temp_true = y_true
            temp_pre = y_pre
            temp_epoch = epoch
            print('“\nThis is the  classification report:...\n')
            print(classification_report(y_true, y_pre, digits=4, target_names=target_names), '\n')
            
            best_model_wts = copy.deepcopy(msresnet.state_dict())
            torch.save(best_model_wts, './models/multi_scale_cnn')

    print(str(100 * float(temp_test) / num_test_instances)[0:6])


    # sio.savemat('result/changingResnet/TrainLoss_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'train_loss': train_loss})
    # sio.savemat('result/changingResnet/TestLoss_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'test_loss': test_loss})
    # sio.savemat('result/changingResnet/TrainAccuracy_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'train_acc': train_acc})
    # sio.savemat('result/changingResnet/TestAccuracy_' + 'ChangingSpeed_Train' + str(100*float(temp_train)/num_train_instances)[0:6] + 'Test' + str(100*float(temp_test)/num_test_instances)[0:6] + '.mat', {'test_acc': test_acc})
    # print(str(float(temp_test)/num_test_instances)[0:6])
    # print(str(float(correct_test)/num_test_instances)[0:6])
    #
    # plt.plot(train_loss)
    # plt.show()

