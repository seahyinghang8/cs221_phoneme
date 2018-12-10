import torch
import torch.nn as nn
import logging

class baseCNN(nn.Module):
    def __init__(self):
        # initializes the model and its weights
        # pass the config in as required
        super(baseCNN, self).__init__()

    def _init_weights(self):
        # perform He initialization
        modules_to_initialize = ['Conv2d', 'Linear']
        for m in self.modules():
            classname = m.__class__.__name__
            if classname in modules_to_initialize:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        logging.debug('Initializing bias {0}.{1} with zeros.'.format(
                            classname, name))
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        logging.debug('Initializing weight {0} using He initialization.'.format(
                            m))
                        nn.init.kaiming_normal_(param)

class cnn2Layer(baseCNN):
    # Very simple CNN. 2 layers of convolution and 2 fully-connected layer
    def __init__(self, num_channels, num_frames, num_classes, num_dimensions):
        # initializes the model and its weights
        super(cnn2Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(num_channels, 3, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(3, 5, kernel_size=(5, 5), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(5)

        # HARD CODED num inputs
        input_size = {
            13: 80,
            26: 240,
            40: 440,
        }

        self.fc1 = nn.Linear(input_size[num_dimensions], 512)
        self.fc2 = nn.Linear(512, num_classes)

        # softmax
        self.softmax = nn.LogSoftmax(dim=1)

        # initializes the weights
        self._init_weights()



    def forward(self, x):
        # returns a tuple of outputs
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.softmax(x)

        return x


class cnn5Layer(baseCNN):
    # Intermediate CNN. 5 layers of convolution and 3 fully-connected layer
    def __init__(self, num_channels, num_frames, num_classes, num_dimensions):
        # initializes the model and its weights
        super(cnn5Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(num_channels, 3, kernel_size=(5, 3), stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(3, 5, kernel_size=(5, 3), stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(5)
        self.conv3 = nn.Conv2d(5, 7, kernel_size=(5, 3), stride=1, padding=2)
        self.conv4 = nn.Conv2d(7, 5, kernel_size=(5, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(5)
        self.conv5 = nn.Conv2d(5, 3, kernel_size=(5, 3), stride=1, padding=1)

        # HARD CODED num inputs
        input_size = {
            13: 162,
            26: 234,
            40: 324,
        }

        self.fc1 = nn.Linear(input_size[num_dimensions], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        # softmax
        self.softmax = nn.LogSoftmax(dim=1)

        # initializes the weights
        self._init_weights()



    def forward(self, x):
        # returns a tuple of outputs
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.relu(self.conv5(x))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.softmax(x)

        return x



class cnn10Layer(baseCNN):
    # Intermediate CNN. 10 layers of convolution and 3 fully-connected layer
    def __init__(self, num_channels, num_frames, num_classes, num_dimensions):
        # initializes the model and its weights
        super(cnn10Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(num_channels, 3, kernel_size=(5, 3), stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(3, 5, kernel_size=(5, 3), stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(5)
        self.conv3 = nn.Conv2d(5, 7, kernel_size=(5, 3), stride=1, padding=2)
        self.conv4 = nn.Conv2d(7, 7, kernel_size=(5, 3), stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(7)
        self.conv5 = nn.Conv2d(7, 9, kernel_size=(5, 3), stride=1, padding=2)
        self.conv6 = nn.Conv2d(9, 7, kernel_size=(5, 3), stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(7)
        self.conv7 = nn.Conv2d(7, 5, kernel_size=(5, 3), stride=1, padding=2)
        self.conv8 = nn.Conv2d(5, 5, kernel_size=(5, 3), stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(5)
        self.conv9 = nn.Conv2d(5, 3, kernel_size=(5, 3), stride=1, padding=1)
        self.conv10 = nn.Conv2d(3, 5, kernel_size=(5, 3), stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(5)

        # HARD CODED num inputs
        input_size = {
            13: 340,
            26: 420,
            40: 520,
        }

        self.fc1 = nn.Linear(input_size[num_dimensions], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        # softmax
        self.softmax = nn.LogSoftmax(dim=1)

        # initializes the weights
        self._init_weights()



    def forward(self, x):
        # returns a tuple of outputs
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.bn6(x)
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.bn8(x)
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.bn10(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.softmax(x)

        return x