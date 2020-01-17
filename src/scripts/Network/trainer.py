import sys
sys.path.append('../.')
import glob
import torch
import JSONHelper
from model import *
from config import *
import torch.nn as nn
import torch.optim as optim
import dataset_loader as dataloader
from torch.autograd import Variable
import torch.utils.data as torchdata


class Trainer:
    def __init__(self, train_list):
        self.dataset_train = dataloader.DatasetLoad(train_list)

    def train(self):
        self.dataloader_train = torchdata.DataLoader(self.dataset_train)
        # self.dataloader_train = torchdata.DataLoader(self.dataset_train, batch_size=4, shuffle=True,
        #                                              num_workers=2, drop_last=False)

        model = Net(1, 1).cpu()
        distance = nn.MSELoss()
        # distance = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

        # ----------------------------------------------------------------------
        for epoch in range(num_epochs):
            running_loss = 0.0

            for idx, sample in enumerate(self.dataloader_train):
                input = Variable(sample['occ_grid']).cpu()
                target = Variable(sample['occ_gt']).cpu()

                # zero the parameter gradients
                optimizer.zero_grad()

                # ===================forward=====================
                output = model(input)
                loss = distance(output, target)
                # ===================backward + optimize====================
                loss.backward()
                optimizer.step()

                # ===================log========================
                running_loss += loss.item()
                if idx % 50 == 49:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 50))
                    running_loss = 0.0
        print("Finished training")

        # Save the trained model
        torch.save(model.state_dict(), '/home/parika/WorkingDir/complete3D/Assets/output-network/model.pth')


if __name__ == '__main__':
    synset_lst = ['02747177', '02801938', '02773838', '02933112', '02942699', '02946921', '03636649']
    train_list = []

    params = JSONHelper.read("../parameters.json")

    for synset_id in synset_lst:
        for f in glob.glob(params["shapenet_raytraced"] + synset_id + "/*.txt"):
            model_id = f[f.rfind('/')+1:f.rfind('.')]
            train_list.append({'synset_id': synset_id, 'model_id': model_id})
    # train_list = []
    # train_list.append({'synset_id': '02933112', 'model_id': '2f0fd2a5e181b82a4267f85fb94fa2e7'})
    # train_list.append({'synset_id': '02933112', 'model_id': 'a46d947577ecb54a6bdcd672c2b17215'})
    # train_list.append({'synset_id': '02933112', 'model_id': '37ba0371250bcd6de117ecc943aca233'})
    # train_list.append({'synset_id': '02933112', 'model_id': 'bd2bcee265b1ee1c7c373e0e7470a338'})
    # train_list.append({'synset_id': '02933112', 'model_id': '8a2aadf8fc4f092c5ee1a94f1da3a5e'})
    #
    # train_list.append({'synset_id': '02942699', 'model_id': '6d036fd1c70e5a5849493d905c02fa86'})
    # train_list.append({'synset_id': '02942699', 'model_id': '97690c4db20227d248e23e2c398d8046'})
    # train_list.append({'synset_id': '02942699', 'model_id': 'e9e22de9e4c3c3c92a60bd875e075589'})
    # train_list.append({'synset_id': '02942699', 'model_id': '51176ec8f251800165a1ced01089a2d6'})
    # train_list.append({'synset_id': '02942699', 'model_id': '46c09085e451de8fc3c192db90697d8c'})
    #
    # train_list.append({'synset_id': '02946921', 'model_id': 'ebcbb82d158d68441f4c1c50f6e9b74e'})
    # train_list.append({'synset_id': '02946921', 'model_id': '3703ada8cc31df4337b00c4c2fbe82aa'})
    # train_list.append({'synset_id': '02946921', 'model_id': 'fd40fa8939f5f832ae1aa888dd691e79'})
    # train_list.append({'synset_id': '02946921', 'model_id': '3fd8dae962fa3cc726df885e47f82f16'})
    # train_list.append({'synset_id': '02946921', 'model_id': 'b1980d6743b7a98c12a47018402419a2'})
    #
    # train_list.append({'synset_id': '03636649', 'model_id': 'bde9b62e181cd4694fb315ce917a9ec2'})
    # train_list.append({'synset_id': '03636649', 'model_id': '967b6aa33d17c109e81edb73cdd34eeb'})
    # train_list.append({'synset_id': '03636649', 'model_id': '6ffb0636180aa5d78570a59d0416a26d'})
    # train_list.append({'synset_id': '03636649', 'model_id': 'f449dd0eb25773925077539b37310c29'})
    # train_list.append({'synset_id': '03636649', 'model_id': '989694b21ed5752d4c61a7cce317bfb7'})
    #

    print("Training data size: ", len(train_list))
    trainer = Trainer(train_list)
    trainer.train()
