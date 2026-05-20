import os
from model.LTLNet import *



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, model_name, dataset_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        if model_name == 'LTLNet':
            if mode == 'train':
                if dataset_name == 'NUDT':
                    self.model = LTLNET(dataset_name='NUDT', mode='train')
                elif dataset_name == 'NUAA':
                    self.model = LTLNET(dataset_name='NUAA', mode='train')
                elif dataset_name == 'IRSTD':
                    self.model = LTLNET(dataset_name='IRSTD', mode='train')
                else:
                    self.model = LTLNET(dataset_name='other', mode='train')
            else:
                if dataset_name == 'NUDT':
                    self.model = LTLNET(dataset_name='NUDT', mode='test')
                elif dataset_name == 'NUAA':
                    self.model = LTLNET(dataset_name='NUAA', mode='test')
                elif dataset_name == 'IRSTD':
                    self.model = LTLNET(dataset_name='IRSTD', mode='test')
                else:
                    self.model = LTLNET(dataset_name='other', mode='test')

    def forward(self, img, mask=None):
        return self.model(img,mask)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
