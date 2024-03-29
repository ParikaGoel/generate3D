import sys

sys.path.append('../.')
import os
import glob
import torch
import config
import pathlib
import losses
import JSONHelper
import numpy as np
import voxel_grid
import torch.nn as nn
import torch.optim as optim
from model_proj_layer import *
import dataset_loader as dataloader
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter
from perspective_projection import ProjectionHelper

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def train(self, epoch):
        self.model.train()
        batch_loss = 0.0

        proj_img_out = os.path.join(params["network_output"], "proj_net/" + synset_id + "/data/proj_imgs/run_%02d"%epoch)
        gt_img_out = os.path.join(params["network_output"], "proj_net/" + synset_id + "/data/gt_imgs/run_%02d"%epoch)
        occ_out = os.path.join(params["network_output"], "proj_net/" + synset_id + "/data/occ/run_%02d"%epoch)
        pathlib.Path(proj_img_out).mkdir(parents=True, exist_ok=True)
        pathlib.Path(gt_img_out).mkdir(parents=True, exist_ok=True)
        pathlib.Path(occ_out).mkdir(parents=True, exist_ok=True)

        for idx, sample in enumerate(self.dataloader_train):
            occ_input = sample['occ_grid'].to(self.device)
            imgs_gt = sample['imgs_gt'].to(self.device)
            poses = sample['poses'].to(self.device)

            poses = poses[:, 0, :, :].unsqueeze(0)
            imgs_gt = imgs_gt[:, 0, :].unsqueeze(0)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # ===================forward=====================
            projection_helper = ProjectionHelper()
            occ, proj_imgs = self.model(occ_input, poses)

            for img_idx, proj_img in enumerate(proj_imgs[0]):
                projection_helper.save_projection(os.path.join(proj_img_out,"img_%02d.png" % img_idx), proj_img)

            for img_idx, img_gt in enumerate(imgs_gt[0]):
                projection_helper.save_projection(os.path.join(gt_img_out, "img_%02d.png" % img_idx), img_gt)

            dataloader.save_sample(os.path.join(occ_out,"pred_occ.txt"), torch.nn.Sigmoid()(occ[0].detach()))
            voxel_grid.txt_to_mesh(os.path.join(occ_out,"pred_occ.txt"), os.path.join(occ_out,"pred_occ.ply"))

            loss = losses.proj_loss(proj_imgs, imgs_gt, self.device)

            # ===================backward + optimize====================
            loss.backward()
            self.optimizer.step()

            # ===================log========================
            batch_loss += loss.item()

            print('Training : [%d : %5d] loss: %.3f' % (epoch + 1, idx + 1, loss.item()))

        train_loss = batch_loss / (idx + 1)
        return train_loss

    def validate(self, epoch):
        self.model.eval()
        batch_loss = 0.0

        with torch.no_grad():
            for idx, sample in enumerate(self.dataloader_val):
                occ_input = sample['occ_grid'].to(self.device)
                imgs_gt = sample['imgs_gt'].to(self.device)
                poses = sample['poses'].to(self.device)

                # ===================forward=====================
                occ, proj_imgs = self.model(occ_input, poses)
                loss = losses.proj_loss(proj_imgs, imgs_gt, self.device)

                # ===================log========================
                batch_loss += loss.item()

                print('Validation : [%d : %5d] loss: %.3f' % (
                epoch + 1, idx + 1, loss.item()))
            val_loss = batch_loss / (idx + 1)
            return val_loss

    def start(self, train_writer, val_writer):
        print("Start training")
        prev_val_loss = 50000
        for epoch in range(100):
            train_loss = self.train(epoch)
            val_loss = self.validate(epoch)
            print("Train loss: %.3f" % train_loss)
            print("Val loss: %.3f" % val_loss)
            train_writer.add_scalar("loss", train_loss, epoch + 1)
            val_writer.add_scalar("loss", val_loss, epoch + 1)

            # Save the trained model
            if val_loss < prev_val_loss:
                print("Saving the model at epoch %d" % epoch)
                torch.save(self.model.state_dict(), params["network_output"] + "proj_net/" + synset_id + "/saved_models/proj.pth")
                prev_val_loss = val_loss

        print("Finished training")
        train_writer.close()
        val_writer.close()

if __name__ == '__main__':
    train_list = []

    for f in glob.glob(params["shapenet_raytraced"] + synset_id + "/*.txt"):
        model_id = f[f.rfind('/') + 1:f.rfind('.')]
        train_list.append({'synset_id': synset_id, 'model_id': model_id})

    # val_list = train_list[250:330]
    # train_list = train_list[:250]
    val_list = train_list[:1]
    train_list = train_list[:1]

    # model_id = "501154f25599ee80cb2a965e75be701c"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training data size: ", len(train_list))
    print("Validation data size: ", len(val_list))
    print("Device: ", device)
    print("Training list: ", train_list)

    train_writer_path = params["network_output"] + "proj_net/" + synset_id + "/logs/logs_proj/train/"
    val_writer_path = params["network_output"] + "proj_net/" + synset_id + "/logs/logs_proj/val/"

    pathlib.Path(train_writer_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_writer_path).mkdir(parents=True, exist_ok=True)

    train_writer, val_writer = create_summary_writers(train_writer_path, val_writer_path)

    trainer = Trainer(train_list, val_list, device)
    trainer.start(train_writer, val_writer)
