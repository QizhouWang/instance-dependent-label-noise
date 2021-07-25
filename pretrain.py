import time, datetime, sys, argparse, numpy

import torch
import torch.optim as optim
import torchvision.models as models

from dataloader import DataLoaderX
from cloth1m import *

# PARSER
# -------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Tackling Instance-Dependent Label Noise via a Universal Probabilistic Model')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='number of epochs to train')
parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train')

parser.add_argument('--weight_decay', '--wd', default=1e-3, type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--model_lr', default=5e-3, type=float, help='SGD lr')
parser.add_argument('--model_step', default=5, type=float, help='SGD scheduler step')
parser.add_argument('--gamma', default=0.1, type=float, help='SGD scheduler lr decay')

momentum, weight_decay = args.momentum, args.weight_decay
model_lr, model_step, gamma = args.model_lr, args.model_step, args.gamma
# -------------------------------------------------------- #

# DATASET
# -------------------------------------------------------- #
print('\n => loading dataset...')                          #
NUM_CLASSES = 14                                           #
trainset = Cloth1M('./Clothing1M', mode = 'train', transform = transform_train)
validset = Cloth1M('./Clothing1M', mode = 'valid', transform = transform_valid)
trainloader = DataLoaderX(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
validloader = DataLoaderX(validset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
# -------------------------------------------------------- #

# MODEL
# -------------------------------------------------------- #
print('\n => constructing model...')                       #
# + Classifier                                             #
model = models.resnet50(num_classes = NUM_CLASSES).to('cuda')
#model.load_state_dict(torch.load('./imagenet_resnet50.pth'))
optimizer = optim.SGD(model.parameters(), lr = model_lr, momentum = momentum, weight_decay = weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, model_step, gamma)
# -------------------------------------------------------- #

# UTILS
# -------------------------------------------------------- #
# + Onehot                                                 #
eye = torch.eye(NUM_CLASSES).to('cuda')                    #
# + Logging                                                #
def logs(msg):                                             #
    clearline = '\b' * (len(msg) + 5)                      #
    sys.stdout.write(clearline + msg)                      #
# -------------------------------------------------------- # 

# TRAIN
for epoch in range(args.epochs):

    model.train()
    time_start = time.time()
    if epoch != 0: scheduler.step()
    correct, total = 0, 0

    for IDX, (inputs, targets, indies, _) in enumerate(trainloader):

        inputs, targets_, indies = inputs.to('cuda'), targets.to('cuda'), indies.to('cuda')
    
        optimizer.zero_grad()
        outputs = t.log_softmax(model(inputs), 1)
        targets = eye[targets_]
        loss = -(outputs * targets).sum(1).mean()
        loss.backward()
        optimizer.step()

        predicts = outputs.detach().argmax(1)
        correct += (predicts == targets_).float().sum().item()
        total += inputs.size(0)

        time_res = datetime.timedelta(seconds = time.time()-time_start).__str__()[:-7]
        logs(('EPOCH %d (%d/%d) | loss %.2f | accu (noise) %.2f%% | '
            % (epoch, IDX + 1, len(trainloader), loss.item(), correct * 100 / total))
            + time_res)

torch.save(model.state_dict(), './pretrained_noise.pth')