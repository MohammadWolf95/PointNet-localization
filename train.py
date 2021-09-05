from LocalizationDataLoader import LocalizationDataLoader
import torch
import torch.nn as nn
from Pointnet_Pointnet2_pytorch.models.pointnet_utils import STN3d
import logging
from tqdm import tqdm
from Pointnet_Pointnet2_pytorch import provider
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

DATASET_DIR = '/data/dataset_laser/sequences/'
LABELS_ARRAY = 'output_shared.npy'
#LABELS_ARRAY = 'output00.npy'
batch_size = 8
train_dataset = LocalizationDataLoader(DATASET_DIR, LABELS_ARRAY)

learning_rate = 0.001
decay_rate=1e-4
epochs = 200
global_epoch = 0
global_step = 0

classes = 131
#classes = 12878 #shared

trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)

writer.close()

class PointNet_localization(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNet_localization, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(classes)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, classes)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
            
    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x



class get_loss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target):
        loss = nn.CrossEntropyLoss()
        output = loss(pred, target)
        #loss = F.nll_loss(pred, target)
        #mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        #total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return output

logger = logging.getLogger("Model")

classifier = PointNet_localization().cuda()

criterion = get_loss().cuda()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=decay_rate
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

def log_string(str):
    logger.info(str)
    print(str)


for epoch in range(0, epochs):
    log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, epochs))
    classifier = classifier.train()
    scheduler.step()
    mean_correct = []
    for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        optimizer.zero_grad()
        classifier = classifier.train()
        scheduler.step()
        
        points = points.data.numpy()
        
        #print('target = %d' %(target))
        #print('target = %d, %d' %(target[0], target[1]))

        
        #print(points.shape)
        #points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        
        #for item in points:
        #    print(item)
        
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        
        #for item in points:
        #    print(item)

        #if not args.use_cpu:
        points, target = points.cuda(), target.cuda()
        pred = classifier(points)
        loss = criterion(pred, target.long())
        writer.add_scalar("Loss/train", loss, epoch, global_step)
        pred_choice = pred.data.max(1)[1]
        
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        loss.backward()
        optimizer.step()
        global_step += 1
        
    train_instance_acc = np.mean(mean_correct)
    log_string('Train Instance Accuracy: %f' % train_instance_acc)
    
writer.flush()

writer.close()