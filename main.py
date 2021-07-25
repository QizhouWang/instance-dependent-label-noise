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

parser.add_argument('--eta_lr', default=5e-2, type=float)
parser.add_argument('--eta_init', default=0.01, type=float)

momentum, weight_decay = args.momentum, args.weight_decay
model_lr, model_step, gamma = args.model_lr, args.model_step, args.gamma
eta_lr, eta_init = args.eta_lr, args.eta_init
# -------------------------------------------------------- #

'''
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

print('----------------------\nparameters\n model\n |- lr %.2e \n |- decay %d \n |- gamma %.2f \n eta\n |- lr %.2e \n |- init %.2e \n----------------------'
% (model_lr, model_step, gamma, eta_lr, eta_init))
'''

# DATASET
# -------------------------------------------------------- #
print('\n => loading dataset...')                          #
NUM_CLASSES = 14
trainset = Cloth1M('./Clothing1M', mode = 'train', transform = transform_train)
validset = Cloth1M('./Clothing1M', mode = 'valid', transform = transform_valid)
trainloader = DataLoaderX(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
validloader = DataLoaderX(validset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
# -------------------------------------------------------- #

# UTILS
# -------------------------------------------------------- #
# + Onehot                                                 #
eye = torch.eye(NUM_CLASSES).to('cuda')                    #
# + Class-conditional probability                          #
def predict_truelabel(etas, targets, pnl):                 #
# re-estimated posterior of true label given eta and posterior of noisy labels 
    part1 = eye[targets.long()] * (1 - etas).view(-1, 1)   #
    part2 = (pnl*etas).view(-1, 1)                         #
    return (part1 + part2).clamp(min = 1e-5).log()         #
# + Logging                                                #
def logs(msg):                                             #
    sys.stdout.write('\r' + msg)                           #
# + Data Imbalance                                         #
def upsample(targets):                                     #
# Clothing1M has a several class-imbalance problem. Note that, the weights can
# be tuned for better experimental results.                #
    weights = t.Tensor([1,1,1,2,5,                         #
                        1,1,1,1,1,                         #
                        2,1,1.5,1,]).to('cuda')            #
    return weights[targets]                                #
# -------------------------------------------------------- # 

# Posterior of Noisy Labels
# -------------------------------------------------------- #
# loading model parameters directly trained on noisy labeled data. 
model = models.resnet50(num_classes = NUM_CLASSES).to('cuda')
model.load_state_dict(torch.load('./pretrained_noise.pth'))
with torch.no_grad():                                      #
    sums, total = 0, 0                                     #
    for IDX, (inputs, targets, indies, _) in enumerate(trainloader):
        inputs, targets, indies = inputs.to('cuda'), targets.to('cuda'), indies.tolist()
        outputs = torch.softmax(model(inputs), 1)          #
        psi = (outputs * eye[targets]).sum(1).cpu().tolist()
        for idx, p in zip(indies, psi):                    #
            trainset.update_psi(idx, p)                    #
        sums += sum(psi)                                   #
        total += len(psi)                                  #
        logs('preprocess (%d/%d) | mean %.2f' % (IDX, len(trainloader), sums/total))
# -------------------------------------------------------- # 


# MODEL
# -------------------------------------------------------- #
print('\n => constructing model...')                       #
# + Classifier                                             #
model = models.resnet50(num_classes = NUM_CLASSES).to('cuda')
#model.load_state_dict(torch.load('./imagenet_resnet50.pth'))
# + Eta                                                    #
ETA = torch.zeros((len(trainset),)).to('cuda') + eta_init  #
# + Optimizer                                              #
optimizer = optim.SGD(model.parameters(), lr = model_lr, momentum = momentum, weight_decay = weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, model_step, gamma)
# -------------------------------------------------------- #



# TRAIN
for epoch in range(args.epochs):
    model.train()
    time_start = time.time()
    if epoch != 0: scheduler.step()
    correct, total = 0, 0
    #eta_hist = torch.Tensor([0] * 10).to('cuda')
    
    for IDX, (inputs, targets, indies, pnl) in enumerate(trainloader):

        inputs, targets, indies, pnl = inputs.to('cuda'), targets.to('cuda'), indies.to('cuda'), pnl.to('cuda').float()
    
        optimizer.zero_grad()
        outputs = t.log_softmax(model(inputs), 1)

        # ALTERNATING OPTIMIZATION
        # ---------------------------------------------------- #
        # + Prediction                                         #
        pyz_x = (predict_truelabel(ETA[indies], targets, pnl) + outputs.detach()).exp()
        pz_x = pyz_x / pyz_x.sum(1).view(-1, 1)                #
        # + Optimization                                       #
        # |- classifier                                        #
        loss = -(upsample(targets) * (pz_x * outputs).sum(1)).mean()
        loss.backward()                                        #
        optimizer.step()                                       #
        # |- confusing                                         #
        # For the simplicty of the updating rule, we actually assume pnl is close
        # to 1. Directly assuming pnl=1 can lead to similar results in practice. 
        if epoch != 0:                                         #
            disparities = (pz_x * (1 + (pnl * ETA[indies] - ETA[indies] - 1).view(-1,1) * eye[targets])).sum(1)
            ETA[indies] += eta_lr * disparities / ETA[indies].clamp(min = 1e-5)
            ETA[indies] = ETA[indies].clamp(min = 0, max = 1)  #
        # ---------------------------------------------------- #

        # ---------------------------------------------------- #
        # + Classifier                                         #
        predicts = outputs.detach().argmax(1)                  #
        correct += (predicts == targets).float().sum().item()  #
        total += inputs.size(0)                                #
        # + Etas                                               #
        etas = ETA[indies]                                     #
        eta_hist += torch.eye(10).to('cuda')[(etas * 10).long().clamp(min = 0, max = 9)].sum(0)
        # ---------------------------------------------------- #
        time_res = datetime.timedelta(seconds = time.time()-time_start).__str__()[:-7]
        logs(('EPOCH %d (%d/%d) | loss %.2f | accu (noise) %.2f%% | '
            % (epoch, IDX + 1, len(trainloader), loss.item(), correct * 100 / total))
            + time_res)                                        #
    log_eta = '\n | eta'                                       #
    for i, d in enumerate(eta_hist.tolist()):                  #
        log_eta += ' %d:%.1e' % (i, d)                         #
    print(log_eta)                                             #
    # -------------------------------------------------------- #

    # VALIDATION
    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        for IDX, (inputs, targets, _, _) in enumerate(validloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            predicts = model(inputs).argmax(1)
            correct += (targets == predicts).float().sum().item()
            total += predicts.size(0)
            logs('VALID %d (%d/%d) | accu %.2f%%'
                % (epoch, IDX + 1, len(validloader), correct * 100 / total)) 
    print('')
