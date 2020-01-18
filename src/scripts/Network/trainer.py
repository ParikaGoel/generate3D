import sys

sys.path.append('../.')
import glob
import torch
import config
import JSONHelper
from model import *
import torch.nn as nn
import torch.optim as optim
import dataset_loader as dataloader
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter

params = JSONHelper.read("../parameters.json")


def create_summary_writers(train_writer_path, val_writer_path):
    """
    :param train_writer_path: Path to the train writer
    :param val_writer_path: Path to the val writer
    :return: Summary writer objects
    """
    writer_train = SummaryWriter(train_writer_path)
    writer_val = SummaryWriter(val_writer_path)
    return writer_train, writer_val


class Trainer:
    def __init__(self, train_list, val_list, device):
        self.dataset_train = dataloader.DatasetLoad(train_list)
        self.dataloader_train = torchdata.DataLoader(self.dataset_train, batch_size=50, shuffle=True,
                                                     num_workers=2, drop_last=False)

        self.dataset_val = dataloader.DatasetLoad(val_list)
        self.dataloader_val = torchdata.DataLoader(self.dataset_val)

        self.device = device
        self.model = Net(1, 1).to(device)

    def loss_and_optimizer(self):
        # self.criterion = nn.MSELoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=1e-5)
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    def train(self, epoch):
        self.model.train()
        batch_loss = 0.0
        running_loss = 0.0

        for idx, sample in enumerate(self.dataloader_train):
            input = sample['occ_grid'].to(self.device)
            target = sample['occ_gt'].to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # ===================forward=====================
            output = self.model(input)
            loss = self.criterion(output, target)
            # ===================backward + optimize====================
            loss.backward()
            self.optimizer.step()

            # ===================log========================
            batch_loss += loss.item()
            running_loss += loss.item()

            if idx % 5 == 4:
                print('Training : [%d : %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 5))
                running_loss = 0.0

        train_loss = batch_loss / (idx + 1)
        return train_loss

    def validate(self):
        self.model.eval()
        batch_loss = 0.0

        with torch.no_grad():
            for idx, sample in enumerate(self.dataloader_val):
                input = sample['occ_grid'].to(self.device)
                target = sample['occ_gt'].to(self.device)

                # ===================forward=====================
                output = self.model(input)
                loss = self.criterion(output, target)

                # ===================log========================
                batch_loss += loss.item()
            val_loss = batch_loss / (idx + 1)
            return val_loss

    def start(self, train_writer, val_writer):
        print("Start training")
        for epoch in range(config.num_epochs):
            train_loss = self.train(epoch)
            val_loss = self.validate()
            print("Train loss: %.3f" % train_loss)
            print("Val loss: %.3f" % val_loss)
            train_writer.add_scalar("loss", train_loss, epoch + 1)
            val_writer.add_scalar("loss", val_loss, epoch + 1)

        print("Finished training")
        train_writer.close()
        val_writer.close()

        # Save the trained model
        torch.save(self.model.state_dict(), params["network_output"] + "model.pth")


if __name__ == '__main__':
    synset_train_lst = ['02691156', '02747177', '02773838', '02801938', '02843684', '02933112', '02942699',
                        '04074963', '04099429', '04460130', '04468005'] #, '03938244']
    synset_val_lst = ['02946921', '03636649', '03710193', '03759954', '04554684']

    train_list = []
    val_list =[]

    for synset_id in synset_train_lst:
        for f in glob.glob(params["shapenet_raytraced"] + synset_id + "/*.txt"):
            model_id = f[f.rfind('/') + 1:f.rfind('.')]
            train_list.append({'synset_id': synset_id, 'model_id': model_id})

    for synset_id in synset_val_lst:
        for f in glob.glob(params["shapenet_raytraced"] + synset_id + "/*.txt"):
            model_id = f[f.rfind('/') + 1:f.rfind('.')]
            val_list.append({'synset_id': synset_id, 'model_id': model_id})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training data size: ", len(train_list))
    print("Validation data size: ", len(val_list))
    print("Device: ", device)

    train_writer_path = params["network_output"] + "logs/train"
    val_writer_path = params["network_output"] + "logs/val"

    train_writer, val_writer = create_summary_writers(train_writer_path, val_writer_path)

    trainer = Trainer(train_list, val_list, device)
    trainer.loss_and_optimizer()
    trainer.start(train_writer, val_writer)
