import sys

sys.path.append('../.')
import os
import glob
import torch
import config
import losses
import random
import pathlib
import datetime
import JSONHelper
from model import *
import data_utils as utils
import eval_metric as metric
import dataset_loader as dataloader
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter

params = JSONHelper.read("../parameters.json")

def create_summary_writers(train_writer_path, val_l1_writer_path, iou_writer_path):
    """
    :param train_writer_path: Path to the train writer
    :param val_writer_path: Path to the val writer
    :return: Summary writer objects
    """
    writer_train = SummaryWriter(train_writer_path)
    writer_val_l1 = SummaryWriter(val_l1_writer_path)
    writer_iou = SummaryWriter(iou_writer_path)
    return writer_train, writer_val_l1, writer_iou


class Trainer:
    def __init__(self, train_list, val_list, device):
        self.dataset_train = dataloader.DatasetLoad(train_list)
        self.dataloader_train = torchdata.DataLoader(self.dataset_train, batch_size=config.batch_size, shuffle=True,
                                                     num_workers=2, drop_last=False)

        self.dataset_val = dataloader.DatasetLoad(val_list)
        self.dataloader_val = torchdata.DataLoader(self.dataset_val, batch_size=config.batch_size, shuffle=False,
                                                   num_workers=2, drop_last=False)

        self.device = device
        if config.model_name == 'Net3D':
            self.model = Net3D(1, 1).to(device)
        elif config.model_name == 'UNet3D':
            self.model = UNet3D(1, 1).to(device)

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
            loss = losses.l1(output, target, use_log_transform=True)
            # ===================backward + optimize====================
            loss.backward()
            self.optimizer.step()

            # ===================log========================
            batch_loss += loss.item()

            if (idx + 1) % 10 == 0:
                print('Training : [iter %d / epoch %d] loss: %.3f' % (idx + 1, epoch + 1, loss.item()))

        train_loss = batch_loss / (idx + 1)
        return train_loss

    def validate(self, epoch, output_save):
        self.model.eval()
        batch_loss = 0.0
        batch_iou = 0.0
        vis_save = os.path.join(output_save, "epoch%02d" % (epoch+1))

        n_batches = len(self.dataloader_val)
        with torch.no_grad():
            for idx, sample in enumerate(self.dataloader_val):
                input = sample['occ_grid'].to(self.device)
                target_df = sample['df_gt'].to(self.device)
                names = sample['name']

                # ===================forward=====================
                output_df = self.model(input)
                loss = losses.l1(output_df, target_df, use_log_transform=False)
                iou = metric.iou_df(output_df, target_df, trunc_dist=1.0)

                # ===================log========================
                batch_loss += loss.item()
                batch_iou += iou

                # save the predictions at the end of the epoch
                if (idx + 1) == n_batches:
                    pred_dfs = output_df[:config.n_vis + 1]
                    target_dfs = target_df[:config.n_vis + 1]
                    names = names[:config.n_vis + 1]
                    utils.save_predictions(vis_save, names, pred_dfs=pred_dfs, target_dfs=target_dfs,
                                           pred_occs=None, target_occs=None)

            val_loss = batch_loss / (idx + 1)
            mean_iou = batch_iou / (idx + 1)
            return val_loss, mean_iou

    def start(self, train_writer, val_l1_writer, iou_writer):
        print("Start training")
        best_val_loss = 50000.0
        best_iou = 0.0
        best_val_loss_epoch = 0
        best_iou_epoch = 0
        start_time = datetime.datetime.now()
        output_vis = params["network_output"] + config.synset_id + "/vis/" + config.model_name + "/" + config.gt_type
        output_model = params["network_output"] + config.synset_id + "/models/" + config.model_name + "/" + config.gt_type
        pathlib.Path(output_vis).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_model).mkdir(parents=True, exist_ok=True)

        for epoch in range(config.num_epochs):
            train_loss = self.train(epoch)
            val_loss_l1, iou = self.validate(epoch, output_vis)
            print("Train loss: %.3f" % train_loss)
            print("Val loss: %.3f" % val_loss_l1)
            print("IOU: %.3f" % iou)
            train_writer.add_scalar("loss", train_loss, epoch + 1)
            val_l1_writer.add_scalar("loss", val_loss_l1, epoch + 1)
            iou_writer.add_scalar("iou", iou, epoch + 1)

            if val_loss_l1 < best_val_loss:
                best_val_loss = val_loss_l1
                best_val_loss_epoch = epoch

            if iou > best_iou:
                best_iou = iou
                best_iou_epoch = epoch

            print("Epoch ", epoch+1 , " finished\n")

            if epoch > 19:
                torch.save(self.model.state_dict(), output_model + "/%02d.pth" % (epoch+1))

        end_time = datetime.datetime.now()
        print("Finished training")
        print("Least val loss ", best_val_loss, " at epoch ", best_val_loss_epoch)
        print("Best iou ", best_iou, " at epoch ", best_iou_epoch)
        print("Time taken: ", start_time.strftime('%D:%H:%M:%S'), " to ", end_time.strftime('%D:%H:%M:%S'))
        train_writer.close()
        val_l1_writer.close()
        iou_writer.close()


if __name__ == '__main__':
    train_list = []

    for f in sorted(glob.glob(params["shapenet_raytraced"] + config.synset_id + "/*.txt")):
        model_id = f[f.rfind('/') + 1:f.rfind('.')]
        train_list.append({'synset_id': config.synset_id, 'model_id': model_id})

    val_list = train_list[5400:6740]
    train_list = train_list[:5400]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training data size: ", len(train_list))
    print("Validation data size: ", len(val_list))
    print("Device: ", device)

    log_dir = params["network_output"] + config.synset_id + "/logs/" + config.model_name + "/" + config.gt_type
    train_writer_path = log_dir + "/train/"
    val_l1_writer_path = log_dir + "/val_l1/"
    iou_writer_path = log_dir + "/iou/"

    pathlib.Path(train_writer_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_l1_writer_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(iou_writer_path).mkdir(parents=True, exist_ok=True)

    train_writer, val_l1_writer, iou_writer = create_summary_writers(train_writer_path, val_l1_writer_path, iou_writer_path)

    trainer = Trainer(train_list, val_list, device)
    trainer.start(train_writer, val_l1_writer, iou_writer)
