import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable


def data_load():
    # load data
    data_transform = transforms.Compose([
                transforms.Grayscale(1),
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                
        ])


    # B-mode data loader
    train_data_Bmode = torchvision.datasets.ImageFolder(root='./COVID-US/data/Dataset/train', transform=data_transform)
    train_data_Bmode_loader = torch.utils.data.DataLoader(train_data_Bmode, batch_size=16,  shuffle=False, num_workers=0)
    num_train_instances = len(train_data_Bmode)

    test_data_Bmode = torchvision.datasets.ImageFolder(root='./COVID-US/data/Dataset/test', transform=data_transform)
    test_data_Bmode_loader = torch.utils.data.DataLoader(test_data_Bmode, batch_size=16, shuffle=False, num_workers=0)
    num_test_instances = len(test_data_Bmode)


#     # R1 data loader
#     train_data_R1 = torchvision.datasets.ImageFolder(root='.../Dataset/1/R1/train', transform=data_transform)
#     train_data_R1_loader = torch.utils.data.DataLoader(train_data_R1, batch_size=16, shuffle=False, num_workers=0)

#     test_data_R1 = torchvision.datasets.ImageFolder(root='.../Dataset/1/R1/test', transform=data_transform)
#     test_data_R1_loader = torch.utils.data.DataLoader(test_data_R1, batch_size=16, shuffle=False, num_workers=0)


#     # R4 data loader
#     train_data_R4 = torchvision.datasets.ImageFolder(root='.../Dataset/1/R4/train', transform=data_transform)
#     train_data_R4_loader = torch.utils.data.DataLoader(train_data_R4,  batch_size=16, shuffle=False, num_workers=0)

#     test_data_R4 = torchvision.datasets.ImageFolder(root='.../Dataset/1/R4/test', transform=data_transform)
#     test_data_R4_loader = torch.utils.data.DataLoader(test_data_R4, batch_size=16, shuffle=False, num_workers=0)


#     # S1 data loader
#     train_data_S1= torchvision.datasets.ImageFolder(root='.../Dataset/1/S1/train', transform=data_transform)
#     train_data_S1_loader = torch.utils.data.DataLoader(train_data_S1, batch_size=16, shuffle=False, num_workers=0)

#     test_data_S1 = torchvision.datasets.ImageFolder(root='.../Dataset/1/S1/test', transform=data_transform)
#     test_data_S1_loader = torch.utils.data.DataLoader(test_data_S1, batch_size=16, shuffle=False, num_workers=0)


#     # S4 data loader
#     train_data_S4= torchvision.datasets.ImageFolder(root='.../Dataset/1/S4/train', transform=data_transform)
#     train_data_S4_loader = torch.utils.data.DataLoader(train_data_S4, batch_size=16, shuffle=False, num_workers=0)

#     test_data_S4 = torchvision.datasets.ImageFolder(root='.../Dataset/1/S4/test', transform=data_transform)
#     test_data_S4_loader = torch.utils.data.DataLoader(test_data_S4, batch_size=16, shuffle=False, num_workers=0)


    # do concatenation for training data
    samples_train = []
    labels_train = []
    train_flag = 0

#     for (samples_Bmode, labels_Bmode), (samples_R1, labels_R1), (samples_R4, labels_R4), (samples_S1, labels_S1), (samples_S4, labels_S4) \
#             in zip(train_data_Bmode_loader, train_data_R1_loader,  train_data_R4_loader, train_data_S1_loader, train_data_S4_loader):
    for (samples_Bmode, labels_Bmode) in train_data_Bmode_loader:
        train_flag += 1
        samples1 = Variable(samples_Bmode)
#         samples2 = Variable(samples_R1)
#         samples3 = Variable(samples_R4)
#         samples4 = Variable(samples_S1)
#         samples5 = Variable(samples_S4)
        sample_train = samples1
        labels = labels_Bmode.squeeze()
        label_train = Variable(labels)
        if train_flag == 1:
            samples_train = sample_train
            labels_train = label_train
        if train_flag > 1:
            samples_train = torch.cat([samples_train, sample_train], dim=0)
            labels_train = torch.cat([labels_train, label_train], dim=0) 


    # do concatenation for training data
    samples_test = []
    labels_test = []
    test_flag = 0

#     for (samples_Bmode, labels_Bmode), (samples_R1, labels_R1), (samples_R4, labels_R4), (samples_S1, labels_S1), (samples_S4, labels_S4) \
#             in zip(test_data_Bmode_loader, test_data_R1_loader, test_data_R4_loader, test_data_S1_loader, test_data_S4_loader):
    for (samples_Bmode, labels_Bmode) in test_data_Bmode_loader:
        test_flag += 1
        samples1 = Variable(samples_Bmode)
#         samples2 = Variable(samples_R1)
#         samples3 = Variable(samples_R4)
#         samples4 = Variable(samples_S1)
#         samples5 = Variable(samples_S4)
#         sample_test = torch.cat([samples1, samples2, samples3, samples4, samples5], dim=1)
        sample_test = samples1
        labels = labels_Bmode.squeeze()
        label_test = Variable(labels)
        if test_flag == 1:
            samples_test = sample_test
            labels_test = label_test
        if test_flag > 1:
            samples_test = torch.cat([samples_test, sample_test], dim=0)
            labels_test = torch.cat([labels_test, label_test], dim=0)

    return samples_train, labels_train, samples_test, labels_test, num_train_instances, num_test_instances

