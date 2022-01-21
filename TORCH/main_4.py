import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import caffeine
import numpy as np
from matplotlib import pyplot as plt
from absl import flags
from absl import app
import time

import network_partial
import network_self

FLAGS = flags.FLAGS

flags.DEFINE_integer('MAX_KERNEL', default=3,
    help=('size of the over-parameterized kernel'))
flags.DEFINE_float('threshold_low1', default=0.1,
    help=('the threshold of activating w_5x5/3x3 outer shell'))
flags.DEFINE_float('threshold_high1', default=10,
    help=('the threshold of activating w_7x7 outer shell'))
flags.DEFINE_float('threshold_low2', default=0.1,
    help=('the threshold of activating w_5x5/3x3 outer shell'))
flags.DEFINE_float('threshold_high2', default=10,
    help=('the threshold of activating w_7x7 outer shell'))
flags.DEFINE_float('threshold_low3', default=0.1,
    help=('the threshold of activating w_5x5/3x3 outer shell'))
flags.DEFINE_float('threshold_high3', default=10,
    help=('the threshold of activating w_7x7 outer shell'))
flags.DEFINE_float('threshold_low4', default=0.1,
    help=('the threshold of activating w_5x5/3x3 outer shell'))
flags.DEFINE_float('threshold_high4', default=10,
    help=('the threshold of activating w_7x7 outer shell'))


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#def create_mask(size1,size2,size3,size4):

def main(unused_arg):
    MAX = FLAGS.MAX_KERNEL
    th_low1 = FLAGS.threshold_low1
    th_high1 = FLAGS.threshold_high1
    th_low2 = FLAGS.threshold_low2
    th_high2 = FLAGS.threshold_high2
    th_low3 = FLAGS.threshold_low3
    th_high3 = FLAGS.threshold_high3
    th_low4 = FLAGS.threshold_low4
    th_high4 = FLAGS.threshold_high4
    #net = network.manual_net()
    #t_list = create_t(MAX)
    #print(t_list)
    net = network_self.self_net(MAX)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    start_time = time.time()
    for epoch in range(15):  # loop over the dataset multiple times
        print("epoch in the first network:", epoch +1 )
        running_loss = 0.0
        
        weight1 = net.conv1.weight.data.numpy()
        weight2 = net.conv2.weight.data.numpy()
        weight3 = net.conv3.weight.data.numpy()
        weight4 = net.conv4.weight.data.numpy()
        
        t1_5x5_3 = np.sum(np.sum(np.sum(np.square(weight1[:,:,1:6,1:6]),axis=2),axis=2),axis=1) - np.sum(np.sum(np.sum(np.square(weight1[:,:,2:5,2:5]),axis=2),axis=2),axis=1)
        t2_5x5_3 = np.sum(np.sum(np.sum(np.square(weight2[:,:,1:6,1:6]),axis=2),axis=2),axis=1) - np.sum(np.sum(np.sum(np.square(weight2[:,:,2:5,2:5]),axis=2),axis=2),axis=1)
        t3_5x5_3 = np.sum(np.sum(np.sum(np.square(weight3[:,:,1:6,1:6]),axis=2),axis=2),axis=1) - np.sum(np.sum(np.sum(np.square(weight3[:,:,2:5,2:5]),axis=2),axis=2),axis=1)
        t4_5x5_3 = np.sum(np.sum(np.sum(np.square(weight4[:,:,1:6,1:6]),axis=2),axis=2),axis=1) - np.sum(np.sum(np.sum(np.square(weight4[:,:,2:5,2:5]),axis=2),axis=2),axis=1)
        t1sum = sum(t1_5x5_3)
        t2sum = sum(t2_5x5_3)
        t3sum = sum(t3_5x5_3)
        t4sum = sum(t4_5x5_3)
        print('outer shell values before epoch:',sum(t1_5x5_3),sum(t2_5x5_3),sum(t3_5x5_3),sum(t4_5x5_3))

        if epoch != 0:
            with torch.no_grad():
                if t1sum < th_low1:
                    print("layer1 is shrinking to 3x3 at epoch {}".format(epoch+1))
                    for i in range(7):
                        for j in range(7):
                            if i == 0 or i == 1 or i == 5 or i == 6 or j == 0 or j == 1 or j == 5 or j == 6:
                                net.conv1.weight[:,:,i, j] = 0.
                elif t1sum > th_high1:
                    print("layer1 is expanding to 7x7 at epoch {}".format(epoch+1))
                else:
                    print("layer1 is staying 5x5")
                    for i in range(7):
                        for j in range(7):
                            if i == 0 or i == 6 or j == 0 or j == 6:
                                net.conv1.weight[:,:,i, j] = 0.

                if t2sum < th_low2:
                    print("layer2 is shrinking to 3x3 at epoch {}".format(epoch+1))
                    for i in range(7):
                        for j in range(7):
                            if i == 0 or i == 1 or i == 5 or i == 6 or j == 0 or j == 1 or j == 5 or j == 6:
                                net.conv2.weight[:,:,i, j] = 0.
                elif t2sum > th_high2:
                    print("layer2 is expanding to 7x7 at epoch {}".format(epoch+1))
                else:
                    print("layer2 is staying 5x5")
                    for i in range(7):
                        for j in range(7):
                            if i == 0 or i == 6 or j == 0 or j == 6:
                                net.conv2.weight[:,:,i, j] = 0.

                if t3sum < th_low3:
                    print("layer3 is shrinking to 3x3 at epoch {}".format(epoch+1))
                    for i in range(7):
                        for j in range(7):
                            if i == 0 or i == 1 or i == 5 or i == 6 or j == 0 or j == 1 or j == 5 or j == 6:
                                net.conv3.weight[:,:,i, j] = 0.
                elif t3sum > th_high3:
                    print("layer3 is expanding to 7x7 at epoch {}".format(epoch+1))
                else:
                    print("layer3 is staying 5x5")
                    for i in range(7):
                        for j in range(7):
                            if i == 0 or i == 6 or j == 0 or j == 6:
                                net.conv2.weight[:,:,i, j] = 0.
                     
                if t4sum < th_low4:
                    print("layer4 is shrinking to 3x3 at epoch {}".format(epoch+1))
                    for i in range(7):
                        for j in range(7):
                            if i == 0 or i == 1 or i == 5 or i == 6 or j == 0 or j == 1 or j == 5 or j == 6:
                                net.conv4.weight[:,:,i, j] = 0.
                elif t4sum > th_high4:
                    print("layer4 is expanding to 7x7 at epoch {}".format(epoch+1))
                else:
                    print("layer4 is staying 5x5")
                    for i in range(7):
                        for j in range(7):
                            if i == 0 or i == 6 or j == 0 or j == 6:
                                net.conv2.weight[:,:,i, j] = 0.
            
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            '''
            t1_5x5_3=torch.Tensor(np.sum(np.sum(np.sum(np.square(weight1[:,:,1:4,1:4]),axis=2),axis=2),axis=1))
            t2_5x5_3=torch.Tensor(np.sum(np.sum(np.sum(np.square(weight2[:,:,1:4,1:4]),axis=2),axis=2),axis=1))
            t3_5x5_3=torch.Tensor(np.sum(np.sum(np.sum(np.square(weight3[:,:,1:4,1:4]),axis=2),axis=2),axis=1))
            t4_5x5_3=torch.Tensor(np.sum(np.sum(np.sum(np.square(weight4[:,:,1:4,1:4]),axis=2),axis=2),axis=1))
            t1_5x5=torch.Tensor(np.sum(np.sum(np.sum(np.square(weight1),axis=2),axis=2),axis=1))
            t2_5x5=torch.Tensor(np.sum(np.sum(np.sum(np.square(weight2),axis=2),axis=2),axis=1))
            t3_5x5=torch.Tensor(np.sum(np.sum(np.sum(np.square(weight3),axis=2),axis=2),axis=1))
            t4_5x5=torch.Tensor(np.sum(np.sum(np.sum(np.square(weight4),axis=2),axis=2),axis=1))
            t1_5x5_new -= torch.sigmoid(torch.Tensor(t1_5x5) - torch.Tensor(t1_5x5_3))
            t2_5x5_new -= torch.sigmoid(torch.Tensor(t2_5x5) - torch.Tensor(t2_5x5_3))
            t3_5x5_new -= torch.sigmoid(torch.Tensor(t3_5x5) - torch.Tensor(t3_5x5_3))
            t4_5x5_new -= torch.sigmoid(torch.Tensor(t4_5x5) - torch.Tensor(t4_5x5_3))
            '''
            #print('5x5/3x3 shell before norm:',np.sum(np.sum(np.sum(np.square(weight1),axis=2),axis=2),axis=1))
            #print('5x5/3x3 shell after norm:',np.sum(abs(weight_norm1),axis=0) )
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('epoch: %d, num_update: %5d loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            #end of updating 'for' loop
        
        #t1_5x5_3 = np.sum(np.sum(np.sum(np.square(weight1[:,:,1:6,1:6]),axis=2),axis=2),axis=1) - np.sum(np.sum(np.sum(np.square(weight1[:,:,2:5,2:5]),axis=2),axis=2),axis=1)
        #t2_5x5_3 = np.sum(np.sum(np.sum(np.square(weight2[:,:,1:6,1:6]),axis=2),axis=2),axis=1) - np.sum(np.sum(np.sum(np.square(weight2[:,:,2:5,2:5]),axis=2),axis=2),axis=1)
        #t3_5x5_3 = np.sum(np.sum(np.sum(np.square(weight3[:,:,1:6,1:6]),axis=2),axis=2),axis=1) - np.sum(np.sum(np.sum(np.square(weight3[:,:,2:5,2:5]),axis=2),axis=2),axis=1)
        #t4_5x5_3 = np.sum(np.sum(np.sum(np.square(weight4[:,:,1:6,1:6]),axis=2),axis=2),axis=1) - np.sum(np.sum(np.sum(np.square(weight4[:,:,2:5,2:5]),axis=2),axis=2),axis=1)

        print('outer shell values after an epoch:',sum(t1_5x5_3),sum(t2_5x5_3),sum(t3_5x5_3),sum(t4_5x5_3) )
        #Zeroing the outer shell
        

        
        #weight = net.conv1.weight.data.numpy()
        #weight_norm = net.batch_norm1.weight.data.numpy()
        #print('5x5/3x3 shell after norm:',sum(weight_norm,axis=) )
        #print('5x5/3x3 shell after norm:',sum(sum(weight_norm[0,0,:,:])) )
    print('Finished the First Training')
    print("--- %s seconds ---" % (time.time() - start_time))
    #net_7 = network_partial.wider_net(7,7,7,7)
     
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    #load the checkpoint
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            '''
            model_ft = models.resnet50(pretrained=True)
            ct = 0
            for child in model_ft.children():
                ct += 1
                if ct < 7:
                    for param in child.parameters():
                        param.requires_grad = False
            '''
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the initial network on the 10000 test images: %d %%' % (100 * correct / total))

    ########
    ########
    ######## another network ################################################

    net2 = network_partial.wider_net(5,5,5,5)

    #Zeroing
    #weight1 = net2.conv1.weight.data.numpy()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)
    start_time = time.time()

    for epoch in range(15):  # loop over the dataset multiple times
        print("epoch in the second network:", epoch +1 )
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net2(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            weight1 = net2.conv1.weight.data.numpy()
            weight2 = net2.conv2.weight.data.numpy()
            weight3 = net2.conv3.weight.data.numpy()
            weight4 = net2.conv4.weight.data.numpy()
            
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('epoch: %d, num_update: %5d loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            #end of updating 'for' loop

    print('Finished the First Training')
    print("--- %s seconds ---" % (time.time() - start_time))
    #net_7 = network_partial.wider_net(7,7,7,7)
     
    
    #PATH = './cifar_net.pth'
    PATH_TEMP = './cifar_temp_7.pth'
    torch.save(net2.state_dict(), PATH_TEMP)
    #torch.save(net.state_dict(), PATH_TEMP)
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    #load the checkpoint
    net2.load_state_dict(torch.load(PATH_TEMP))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net2(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the 7777 network on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
  app.run(main)