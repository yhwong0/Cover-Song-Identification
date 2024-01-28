from code_on_shs100k.load_features import split_training_data,load_data
from code_on_shs100k.simple_model import CNN_from_mirex
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from tqdm import tqdm
import gc
import os
import warnings
from torchmetrics.classification import MulticlassAccuracy
from torchsummary import summary
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAveragePrecision
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#warnings.filterwarnings('ignore')

def plot_training_curve(path_to_training_info):
    if path_to_training_info.endswith(".npy"):
        training_info = np.load(path_to_training_info, allow_pickle=True)
        print(training_info.shape)
        print("training info index:",training_info[0])
    else:
            raise IOError("Incorrect file type!")
    training_loss = training_info[1]
    test_loss = training_info[2]
    test_acc = training_info[3]

    # plot 1: training loss and test loss against epochs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1,)
    plt.plot(training_loss, label = "Training Loss")
    plt.plot(test_loss, label = "Test Loss")

    plt.title("Training Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()


    # plot 2: Accuracy against epochs
    plt.subplot(1, 2, 2)
    plt.plot(test_acc,label = "Top 1 accuracy")
    plt.ylim(0, 100)
    print(len(test_acc))

    print("max value of top 1 acc, index", np.max(test_acc), np.argmax(test_acc))
    if(training_info.shape[0]>4):
        test_top_k_acc = training_info[4]

        print("max value of top k acc, index",np.max(test_top_k_acc) ,np.argmax(test_top_k_acc))
        plt.plot(test_top_k_acc,label = "Top 10 accuracy")
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

def load_feature_extractor(model_path):
    "unfinihsed"
    model = CNN_from_mirex()
    model = nn.DataParallel(model)

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

    model.load_state_dict(torch.load(model_path))
    model.eval()




if __name__ == '__main__':
    print("Welcome to the training program")
    #class for data loading
    class Data(Dataset): #referenced from https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2
        def __init__(self, X_train, y_train):
            # need to convert float64 to float32 else
            # will get the following error
            # RuntimeError: expected scalar type Double but found Float
            self.X = torch.from_numpy(X_train.astype(np.float32))
            # need to convert float64 to Long else
            # will get the following error
            # RuntimeError: expected scalar type Long but found Float
            #self.y = torch.from_numpy(y_train).type(torch.LongTensor)
            self.y = y_train.type(torch.LongTensor)
            self.len = self.X.shape[0]

        def __getitem__(self, index):
            return self.X[index], self.y[index]

        def __len__(self):
            return self.len

    #load features
    print("loading features")
    #load_features_from_proposed_dataset()
    #data = load_data("","/research/d2/fyp23/tcor0/cqt_npy",extracted_cqt= True,down_sample = True,slice_cqt = False, all_data_saved_path = "F:\FYP\\results\\1105_normalized_cqt\\test_data_melody_cqt_npy.npy")
    #data = load_data("","C:\FYP_data_test\\106_npy",extracted_cqt= True,down_sample = True,slice_cqt = False)
    #data = load_data("", "C:\\FYP_data_test\\",extracted_cqt= False,down_sample = True)

    plot_training_curve("F:\FYP\\results\\1115melody2000\\training_info.npy")
    exit(123)
    #split_training_data
    _,_,x_test,y_test = split_training_data(data)
    #y_train = torch.from_numpy(y_train.astype(np.int32))
    #y_train = y_train.to(torch.int64)
    y_test = torch.from_numpy(y_test.astype(np.int32))
    y_test = y_test.to(torch.int64)

    #load model and specify optimizer and loss function
    batch_size = 32

    model = CNN_from_mirex()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('F:\FYP\\results\\1105_normalized_cqt\\4_11model_npy_cqt35.pth'))
    model.eval()
    # multi-gpu usage, still investigating

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)

    #load test data with the Data Class and dataloader function

    testdata = Data(x_test, y_test)
    #testdata = Data(x_test,one_hot_y_test)
    testloader = DataLoader(testdata, batch_size=batch_size,shuffle=True, num_workers=0)

    torch.cuda.empty_cache()
    gc.collect()

    training_loss_epochs = []
    testing_loss_epochs = []
    test_accuracy=[]


    # model evaluation from https://saturncloud.io/blog/how-to-test-pytorch-model-accuracy-a-guide-for-data-scientists/
    correct = 0
    total = 0
    running_test_loss = 0.0

    tp=0
    fp=0
    tn=0
    fn=0
    with torch.no_grad():
        total_prediction =torch.empty((0,0), dtype=torch.int64)
        total_target = torch.empty((0,0), dtype=torch.int64)
        count = 0
        for data in testloader:

                inputs, labels = data
                inputs = inputs.unsqueeze(1)
                if torch.cuda.is_available():
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    device = torch.device("cuda:0")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    total_target = total_target.to(device)
                    total_prediction = total_prediction.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if count == 0:
                    total_prediction = outputs
                    total_target = labels
                    count +=1
                else:
                    #print(total_prediction.shape)
                    #print(total_target.shape)
                    total_prediction = torch.cat((total_prediction,outputs),0)
                    total_target = torch.cat((total_target, labels), 0)

        print(total_prediction.shape)
        print(total_target.shape)
        topk = 10
        number_of_class = 917
        acc_metric = Accuracy(task="multiclass",num_classes=number_of_class,top_k =topk).to(device)
        acc = acc_metric(total_prediction, total_target).cpu().numpy()

        precision_metric = MulticlassAveragePrecision(num_classes=number_of_class, average="macro", thresholds=None)
        precision = precision_metric(total_prediction, total_target).cpu().numpy()
        print("The Top "+str(topk)+" accuracy is: " + str(100*acc) + "%")

        print('The Top 1 Accuracy of the network on the test audio '+': %.5f %%' % (100 * correct / total))
        print("The mean Average precision is: " + str(100*precision) + "%")
        test_loss = float(running_test_loss / len(testloader.sampler))
        #print(len(testloader.sampler))
        #testing_loss_epochs.append(test_loss)
        test_accuracy.append((100 * correct / total))
        print("Test loss: ",test_loss)







