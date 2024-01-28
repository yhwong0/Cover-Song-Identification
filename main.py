from code_on_shs100k.load_features import split_training_data,load_data
from code_on_shs100k.simple_model import CNN_from_mirex
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import gc
import os
import warnings
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAveragePrecision
from torchsummary import summary
import datetime


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
#warnings.filterwarnings('ignore')


if __name__ == '__main__':
    print("Welcome to the training program")
    current_time = datetime.date.today()
    print("Current Date: ",current_time)
    date = str(current_time.month)+str(current_time.day)
    torch.backends.cudnn.enabled = False

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
    data = load_data("","/research/d2/fyp23/tcor0/train_ori01",extracted_cqt= True,down_sample = True,slice_cqt = False)#, all_data_saved_path = "/research/d2/fyp23/yhwong0/1115_melody2000/new_all_melody_cqt_term1.npy")
    #data = load_data("","C:\FYP_data_test\\106_npy",extracted_cqt= True,down_sample = True,slice_cqt = False)
    #data = load_data("", "C:\\FYP_data_test\\",extracted_cqt= False,down_sample = True)


    #split_training_data
    x_train,y_train,x_test,y_test = split_training_data(data)
    print("training data shape:",x_train[0].shape)
    y_train = torch.from_numpy(y_train.astype(np.int32))
    y_train = y_train.to(torch.int64)
    y_test = torch.from_numpy(y_test.astype(np.int32))
    y_test = y_test.to(torch.int64)


    #map to one hot encoding vectors:
    #one_hot_y_train = F.one_hot(torch.tensor(y_train.astype(np.int64)))
    #one_hot_y_test = F.one_hot(torch.tensor(y_test.astype(np.int64)))

    #find out all classes in the dataset
    unique_list = []
    for items in y_train:
        if(items not in unique_list):
            unique_list.append(items)
    #make sure the classes [0:t) for t layers in: self.fc1 = nn.Linear(300,t), may need to rearrange the classes for wav in server
    #print("list of class", unique_list)
    print("Number of unique classes:", len(unique_list))

    #exit(123)
    #load model and specify optimizer and loss function
    #temp_model = CNN_from_mirex()

    batch_size = 16
    number_of_class = 917 #instead of 917, discovered during running code
    print("batch size",batch_size)
    print("number of class",number_of_class)

    # print model info
    #print(summary(temp_model, (1, 84, 15000), batch_size=batch_size))  # 128 or 256 when in server


    model = CNN_from_mirex(num_of_classes=number_of_class)
    #multi-gpu usage, still investigating
    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(),lr=lr)
    print("learning rate: ",lr)
    #load train and test data with the Data Class and dataloader function
    traindata = Data(x_train,y_train)
    #traindata = Data(x_train, one_hot_y_train)
    trainloader = DataLoader(traindata, batch_size=batch_size,shuffle=True, num_workers=2)

    testdata = Data(x_test, y_test)
    #testdata = Data(x_test,one_hot_y_test)
    testloader = DataLoader(testdata, batch_size=batch_size,shuffle=True, num_workers=2)

    torch.cuda.empty_cache()
    gc.collect()

    training_loss_epochs = []
    testing_loss_epochs = []
    test_accuracy = []
    test_top_k_accuracy = []
    test_mean_avg_prec = []

    #referenced from https://medium.com/analytics-vidhya/a-simple-neural-network-classifier-using-pytorch-from-scratch-7ebb477422d2
    epochs = 50 #no of epochs
    for epoch in range(epochs):
        loop = tqdm(trainloader,disable=True) #,disable=True
        running_loss = 0.0
        for i, data in enumerate(loop):
        # use tqdm to build a simple progress bar
        # loop = tqdm(loader)
        # print("data ", data)
        #for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            print(data)
            inputs = inputs.unsqueeze(1)

            if torch.cuda.is_available():
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                #print(device)
                #device = torch.device("cuda:0")
                inputs = inputs.to(device)
                model = model.to(device)
                labels = labels.to(device)

            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            #print("length of data:",len(data))
            #print("i ",i)
            #print("input size:",inputs.size())
            running_loss += loss.item()*inputs.size(0) #*data.size(0)
            # add stuff to progress bar in the end, referenced from https://aladdinpersson.medium.com/how-to-get-a-progress-bar-in-pytorch-72bdbf19b35c
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=torch.rand(1).item(), acc=torch.rand(1).item())
        #print(len(trainloader.sampler))
        # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(trainloader.sampler):.5f}')
        training_loss_epochs.append(float(running_loss / len(trainloader.sampler)))
        if(epoch % 1 ==0):
            torch.save(model.state_dict(), date+'_model_org_cqt' + str(epoch) + '.pth')
        # model evaluation from https://saturncloud.io/blog/how-to-test-pytorch-model-accuracy-a-guide-for-data-scientists/
        correct = 0
        total = 0
        running_test_loss = 0.0
        with torch.no_grad():
            total_prediction = torch.empty((0, 0), dtype=torch.int64)
            total_target = torch.empty((0, 0), dtype=torch.int64)
            count = 0
            for data in testloader:
                inputs, labels = data
                inputs = inputs.unsqueeze(1)
                if torch.cuda.is_available():
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    device = torch.device("cuda:0")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                #print("input size:", inputs.size())
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

        topk = 10

        acc_metric = Accuracy(task="multiclass", num_classes=number_of_class, top_k=topk).to(device)
        acc = acc_metric(total_prediction, total_target).cpu().numpy()

        precision_metric = MulticlassAveragePrecision(num_classes=number_of_class, average="macro", thresholds=None)
        precision = precision_metric(total_prediction, total_target).cpu().numpy()
        print("The Top " + str(topk) + " accuracy is: " + str(100 * acc) + "%")

        print('The Top 1 Accuracy of the network on the test audio ' + ': %.5f %%' % (100 * correct / total))
        print("The mean Average precision is: " + str(100 * precision) + "%")
        test_loss = float(running_test_loss / len(testloader.sampler))
        #print(len(testloader.sampler))
        testing_loss_epochs.append(test_loss)
        test_accuracy.append((100 * correct / total))
        test_top_k_accuracy.append(100*acc)
        test_mean_avg_prec.append(100*precision)
        print("Test loss: ",test_loss)
        print("")
        training_info = np.array([["training_loss_epochs","testing_loss_epochs","test_accuracy","test_top_k_accuracy","test_mean_avg_prec"],training_loss_epochs,testing_loss_epochs,test_accuracy,test_top_k_accuracy,test_mean_avg_prec],dtype = object)
        with open('training_info.npy', 'wb') as f:
            np.save(f, training_info)
            f.close()

    torch.save(model, 'model_final' + date +'.pth')




