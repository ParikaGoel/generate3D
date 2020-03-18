import sys

sys.path.append('../.')
import glob
import torch
import config
import pathlib
import losses
import JSONHelper
from model import *
import dataset_loader as dataloader
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter

params = JSONHelper.read("../parameters.json")

synset_id = '04379243'


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
        self.dataloader_train = torchdata.DataLoader(self.dataset_train, batch_size=config.batch_size, shuffle=True,
                                                     num_workers=2, drop_last=False)

        self.dataset_val = dataloader.DatasetLoad(val_list)
        self.dataloader_val = torchdata.DataLoader(self.dataset_val, batch_size=config.batch_size, shuffle=True,
                                                   num_workers=2, drop_last=False)

        self.device = device
        self.model = Net(1, 1).to(device)

    def loss_and_optimizer(self):
        self.criterion = losses.weighted_l1
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def train(self, epoch):
        self.model.train()
        batch_loss = 0.0

        for idx, sample in enumerate(self.dataloader_train):
            input = sample['occ_grid'].to(self.device)
            target = sample['df_gt'].to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # ===================forward=====================
            output = self.model(input)
            loss = self.criterion(self.device, output, target, config.trunc_dist, 2)
            # ===================backward + optimize====================
            loss.backward()
            self.optimizer.step()

            # ===================log========================
            batch_loss += loss.item()

            if (idx + 1) % 10 == 0:
                print('Training : [%d : %5d] loss: %.3f' % (epoch + 1, idx + 1, loss.item()))

        train_loss = batch_loss / (idx + 1)
        return train_loss

    def validate(self):
        self.model.eval()
        batch_loss = 0.0

        with torch.no_grad():
            for idx, sample in enumerate(self.dataloader_val):
                input = sample['occ_grid'].to(self.device)
                target = sample['df_gt'].to(self.device)

                # ===================forward=====================
                output = self.model(input)
                loss = self.criterion(output, target, self.device)

                # ===================log========================
                batch_loss += loss.item()
            val_loss = batch_loss / (idx + 1)
            return val_loss

    def start(self, train_writer, val_writer):
        print("Start training")
        prev_val_loss = 50000.0
        for epoch in range(config.num_epochs):
            train_loss = self.train(epoch)
            val_loss = self.validate()
            print("Train loss: %.3f" % train_loss)
            print("Val loss: %.3f" % val_loss)
            train_writer.add_scalar("loss", train_loss, epoch + 1)
            val_writer.add_scalar("loss", val_loss, epoch + 1)

            if val_loss < prev_val_loss:
                print("Save model on epoch %02d ; val loss: %.3f ; prev val loss: %.3f " % (
                epoch, val_loss, prev_val_loss))
                torch.save(self.model.state_dict(),
                           params["network_output"] + synset_id + "/saved_models/df_trunc.pth")
                prev_val_loss = val_loss

        print("Finished training")
        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    train_list = []

    for f in glob.glob(params["shapenet_raytraced"] + synset_id + "/*.txt"):
        model_id = f[f.rfind('/') + 1:f.rfind('.')]
        train_list.append({'synset_id': synset_id, 'model_id': model_id})

    val_list = train_list[4340:5420]
    train_list = train_list[:4340]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training data size: ", len(train_list))
    print("Validation data size: ", len(val_list))
    print("Device: ", device)

    train_writer_path = params["network_output"] + synset_id + "/logs/logs_df_trunc/train/"
    val_writer_path = params["network_output"] + synset_id + "/logs/logs_df_trunc/val/"

    pathlib.Path(train_writer_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_writer_path).mkdir(parents=True, exist_ok=True)

    train_writer, val_writer = create_summary_writers(train_writer_path, val_writer_path)

    trainer = Trainer(train_list, val_list, device)
    trainer.loss_and_optimizer()
    trainer.start(train_writer, val_writer)
