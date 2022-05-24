import torch.utils.data as Data
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from fusion_models.early_fusion import *
import random
import glob
from torchvision import transforms
import torchvision

torch.manual_seed(2021)  # cpu
torch.cuda.manual_seed(2021)  # gpu
np.random.seed(2021)  # numpy
random.seed(2021)  # random and transforms
torch.backends.cudnn.deterministic = True  

import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
cuda = torch.device('cuda:0')
    
def data_load():
    # load data
    data_transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                
        ])

    test_data_Bmode = torchvision.datasets.ImageFolder(root='./test_data', transform=data_transform)
    test_data_Bmode_loader = torch.utils.data.DataLoader(test_data_Bmode, batch_size=16, shuffle=False, num_workers=0)

    # do concatenation for training data
    samples_test = []
    labels_test = []
    test_flag = 0

    for (samples_Bmode, labels_Bmode) in test_data_Bmode_loader:
        test_flag += 1
        samples1 = Variable(samples_Bmode)
        sample_test = samples1
        labels = labels_Bmode.squeeze()
        label_test = Variable(labels)
        if test_flag == 1:
            samples_test = sample_test
            labels_test = label_test
        if test_flag > 1:
            samples_test = torch.cat([samples_test, sample_test], dim=0)
            labels_test = torch.cat([labels_test, label_test], dim=0)

    return samples_test, labels_test, test_data_Bmode


def worker_init_fn(worker_id):
    np.random.seed(2021 + worker_id)

samples_test, labels_test, test_data_Bmode = data_load()

# data for testing
dataset_test = Data.TensorDataset(samples_test, labels_test)
# identify data loader
dataset_test_loader = Data.DataLoader(dataset=dataset_test, batch_size=60, shuffle=False, num_workers=0,
                                      pin_memory=True, worker_init_fn=worker_init_fn)

if __name__ == '__main__':
    
    msresnet = MSResNet(input_channel=1, layers=[1, 1, 1], num_classes=2)
    msresnet = msresnet.cuda()
    
    msresnet.load_state_dict(torch.load('./models/multi_scale_cnn'))
    msresnet.eval()   # Set model to evaluate mode
    
    for i, (samples, labels) in enumerate(dataset_test_loader):
        samplesV = Variable(samples.cuda())
        samplesX0 = samplesV[:, 0:1, :, :]
        predict_label = msresnet(samplesX0)
        prediction = predict_label.data.max(1)[1]
        prediction = prediction.data.cpu()
        prediction = prediction.numpy()

    f = open('results.txt', 'w')
    for i in range (len(prediction)):
        f.write (f'{test_data_Bmode.samples[i][0][2:]}: {prediction[i]}\n')
    f.close()
    

