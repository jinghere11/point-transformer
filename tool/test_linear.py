import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

class LinearFunc(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(LinearFunc,self).__init__()
        self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn(self.linear(x)))
        # x = self.linear(x)
        # x = x[:,:3].contiguous()
        return x

# input_data = torch.load("x_input.pt")
# label = torch.load("label.pt")
# m = LinearFunc(9391, 3).cuda()
# output = m(input_data)
# p0 = output

# label = torch.ones(output.shape[0]).cuda()
# criterion = nn.CrossEntropyLoss().cuda()
# loss = criterion(output, label.long())
# loss.backward()

# print("OK")


pred = torch.randn((3,6))
values,indices = pred.topk(2,dim=1,sorted=False)
print("OK")







