import torch
import torch.nn as nn
from torch.nn import functional as F


class atten_weig_2D(nn.Module):
    def __init__(self):
        super(atten_weig_2D, self).__init__()


class atten_weig_1D(nn.Module):
    def __init__(self, in_size, out_size, dropout, mu):
        super(atten_weig_1D, self).__init__()
        self.dropout = dropout
        self.in_size = in_size
        self.out_size = out_size
        self.mu = mu
        # local update parameters
        self.local_update_w = nn.Parameter(torch.zeros(1, 1, self.in_size*self.out_size))
        nn.init.xavier_uniform_(self.local_update_w.data, gain=1.414)
        self.local_update_b = nn.Parameter(torch.zeros(1, 1, out_size))
        nn.init.xavier_uniform_(self.local_update_b.data, gain=1.414)
        
        self.bias_GAT = nn.Parameter(torch.zeros(1, 1, in_size*out_size+out_size))
        nn.init.xavier_uniform_(self.bias_GAT.data, gain=1.414)
        # attention parameters
        self.f_1 = nn.Conv1d(1, 1, in_size*out_size+out_size)
        self.f_2 = nn.Conv1d(1, 1, in_size*out_size+out_size)
        
    def forward(self, local_para, neig_para):
        num_neig = neig_para.size(1)
        seq_neig = neig_para # [1, num_nei, sz]

        F_1 = self.f_1(local_para) # [1, 1, 1]
        F_2 = torch.cat([self.f_2(neig_para[:, i, :]) for i in range(num_neig)], 1).unsqueeze(2) #[1,num_nei,1]

        logits = F_1 + torch.transpose(F_2, 1, 2)  # [1, 1, num_neig]
        coefs = F.softmax(F.elu(logits), dim=2)
        # coefs = F.dropout(coefs, self.dropout, training=self.training)

        res1 = torch.matmul(coefs, seq_neig)
        result = F.elu(res1 + self.bias_GAT)  # [1, 1, sz]
        W1 = result[:, :, 0: self.in_size*self.out_size]
        b1 = result[:, :, self.in_size*self.out_size: self.in_size*self.out_size+self.out_size]
        W_add = self.mu*self.local_update_w + (1-self.mu)*W1
        b_add = self.mu*self.local_update_b + (1-self.mu)*b1
        return W_add, b_add


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride): #, neig_weig1, neig_weig2
        super(RestNetBasicBlock, self).__init__()
        self.weig1 = nn.Parameter(torch.zeros(out_channels, in_channels, 3, 3))
        nn.init.xavier_uniform_(self.weig1.data, gain=1.414)
        self.stride = stride
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.weig2 = nn.Parameter(torch.zeros(out_channels, out_channels, 3, 3))
        nn.init.xavier_uniform_(self.weig2.data, gain=1.414)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = F.conv2d(x, self.weig1, bias=None, stride=self.stride, padding=1)
        output = F.relu(self.bn1(output))
        output = F.conv2d(output, self.weig2, bias=None, stride=self.stride, padding=1)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.weig1 = nn.Parameter(torch.zeros(out_channels, in_channels, 3, 3))
        nn.init.xavier_uniform_(self.weig1.data, gain=1.414)
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.weig2 = nn.Parameter(torch.zeros(out_channels, out_channels, 3, 3))
        nn.init.xavier_uniform_(self.weig2.data, gain=1.414)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.weig_extra = nn.Parameter(torch.zeros(out_channels, in_channels, 1, 1))
        nn.init.xavier_uniform_(self.weig_extra.data, gain=1.414)
        self.bn_extra = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        extra_x = self.bn_extra(F.conv2d(x, self.weig_extra, bias=None, stride=self.stride[0], padding=0))
        output = F.conv2d(x, self.weig1, bias=None, stride=self.stride[0], padding=1)
        out = F.relu(self.bn1(output))
        out = F.conv2d(out, self.weig2, bias=None, stride=self.stride[1], padding=1)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class RestNet18_att(nn.Module):
    def __init__(self):
        super(RestNet18_att, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.a_bn1 = nn.BatchNorm2d(8)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = RestNetBasicBlock(8, 8, 1)
        self.layer2 = RestNetBasicBlock(8, 8, 1)

        self.layer3 = RestNetDownBlock(8, 16, [2, 1])
        self.layer4 = RestNetBasicBlock(16, 16, 1)

        self.layer5 = RestNetDownBlock(16, 32, [2, 1])
        self.layer6 = RestNetBasicBlock(32, 32, 1)

        self.layer7 = RestNetDownBlock(32, 64, [2, 1])
        self.layer8 = RestNetBasicBlock(64, 64, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # self.fc = nn.Linear(64, 10)
        self.fc = atten_weig_1D(64, 10, 0.0, 0.9)

    def forward(self, x, local_para, neig_para):
        out = F.elu(self.conv1(x))
        out = self.a_bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        fc_weig, fc_bias = self.fc(local_para, neig_para)
        self.fc_weig = fc_weig.detach().clone()
        self.fc_bias = fc_bias.detach().clone()
        out = F.linear(out, torch.reshape(fc_weig, [10, 64]), torch.reshape(fc_bias, [1, 10]))
        return out
