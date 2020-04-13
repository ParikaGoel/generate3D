import sys

sys.path.append('../.')
import os
import glob
import torch
import config
import losses
import pathlib
import datetime
import argparse
import JSONHelper
from model import *
import data_utils as utils
import eval_metric as metric
import dataset_loader as dataloader
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter

params = JSONHelper.read("../parameters.json")

# python trainer_occ.py --synset_id 04379243 --model_name Net3D --gt_type occ --batch_size 32 --truncation 3

# command line params
parser = argparse.ArgumentParser()
# model params
parser.add_argument('--synset_id', type=str, required=True, help='synset id of the sample category')
parser.add_argument('--model_name', type=str, required=True, help='which model arch to use')
parser.add_argument('--gt_type', type=str, required=True, help='gt representation')
parser.add_argument('--vox_dim', type=int, default=32, help='voxel dim')
# train params
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--save_epoch', type=int, default=10, help='save every model after n epochs')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--decay_lr', type=int, default=10, help='decay learning rate by half every n epochs')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay.')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--n_vis', type=int, default=20, help='number of visualizations to save')

args = parser.parse_args()

def check_model_size(model):
    num_params = 0
    traininable_param = 0
    for param in model.parameters():
        num_params += param.numel()
        if param.requires_grad:
            traininable_param += param.numel()
    print("[Network  Total number of parameters : %.3f M" % (num_params / 1e6))
    print(
        "[Network  Total number of trainable parameters : %.3f M"
        % (traininable_param / 1e6)
    )

def create_summary_writers(train_writer_path, val_bce_writer_path, val_l1_writer_path, iou_writer_path):
    """
    :param train_writer_path: Path to the train writer
    :param val_writer_path: Path to the val writer
    :return: Summary writer objects
    """
    writer_train = SummaryWriter(train_writer_path)
    writer_val_bce = SummaryWriter(val_bce_writer_path)
    writer_val_l1 = SummaryWriter(val_l1_writer_path)
    writer_iou = SummaryWriter(iou_writer_path)
    return writer_train, writer_val_bce, writer_val_l1, writer_iou


class Trainer:
    def __init__(self, train_list, val_list, device):
        self.dataset_train = dataloader.DatasetLoad(data_list=train_list, truncation=args.truncation)
        self.dataloader_train = torchdata.DataLoader(self.dataset_train, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=2, drop_last=False)

        self.dataset_val = dataloader.DatasetLoad(data_list=val_list, truncation=args.truncation)
        self.dataloader_val = torchdata.DataLoader(self.dataset_val, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=2, drop_last=False)

        self.device = device
        if args.model_name == 'Net3D':
            self.model = Net3D(1, 1).to(device)
        elif args.model_name == 'UNet3D':
            self.model = UNet3D(1, 1).to(device)
        else:
            print("Error: Check the model name")
            exit(0)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr, gamma=0.5)

        check_model_size(self.model)

    def train(self, epoch):
        self.model.train()
        batch_loss = 0.0

        for idx, sample in enumerate(self.dataloader_train):
            input = sample['occ_grid'].to(self.device)
            target_occ = sample['occ_gt'].to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # ===================forward=====================
            output_occ = self.model(input)
            loss = losses.bce(output_occ, target_occ)
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
        batch_loss_bce = 0.0
        batch_loss_l1 = 0.0
        batch_iou = 0.0
        vis_save = os.path.join(output_save, "epoch%02d" % (epoch+1))

        n_batches = len(self.dataloader_val)
        with torch.no_grad():
            for idx, sample in enumerate(self.dataloader_val):
                input = sample['occ_grid'].to(self.device)
                target_occ = sample['occ_gt'].to(self.device)
                target_df = sample['occ_df_gt'].to(self.device)
                names = sample['name']

                # ===================forward=====================
                output_occ = self.model(input)
                loss_bce = losses.bce(output_occ, target_occ)

                # Convert occ to df to calculate l1 loss
                output_df = utils.occs_to_dfs(output_occ, trunc=args.truncation, pred=True)
                loss_l1 = losses.l1(output_df, target_df)
                iou = metric.iou_occ(output_occ, target_occ)

                # ===================log========================
                batch_loss_bce += loss_bce.item()
                batch_loss_l1 += loss_l1.item()
                batch_iou += iou

                # save the predictions at the end of the epoch
                if (idx + 1) == n_batches:
                    pred_occs = output_occ[:args.n_vis+1]
                    target_occs = target_occ[:args.n_vis+1]
                    names = names[:args.n_vis+1]
                    utils.save_predictions(vis_save, names, pred_dfs=None, target_dfs=None,
                                           pred_occs=pred_occs, target_occs=target_occs)

            val_loss_bce = batch_loss_bce / (idx + 1)
            val_loss_l1 = batch_loss_l1 / (idx + 1)
            mean_iou = batch_iou / (idx + 1)
            return val_loss_bce, val_loss_l1, mean_iou

    def start(self, train_writer, val_bce_writer, val_l1_writer, iou_writer):
        print("Start training")
        best_val_loss = 50000.0
        best_iou = 0.0
        best_val_loss_epoch = 0
        best_iou_epoch = 0
        iou_at_best_l1 = 0
        l1_at_best_iou = 0
        start_time = datetime.datetime.now()
        output_vis = "%s%s/vis/%s/%s_batch%s_trunc%s" % (params["network_output"], args.synset_id, args.model_name,
                                                         args.gt_type, args.batch_size, args.truncation)

        output_model = "%s%s/models/%s/%s_batch%s_trunc%s" % (params["network_output"], args.synset_id, args.model_name,
                                                              args.gt_type, args.batch_size, args.truncation)


        pathlib.Path(output_vis).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_model).mkdir(parents=True, exist_ok=True)

        for epoch in range(config.num_epochs):
            train_loss = self.train(epoch)
            val_loss_bce, val_loss_l1, iou = self.validate(epoch, output_vis)
            self.scheduler.step()
            print("Train loss: %.3f" % train_loss)
            print("Val loss (bce): %.3f" % val_loss_bce)
            print("Val loss (l1): %.3f" % val_loss_l1)
            print("IOU: %.3f" % iou)
            train_writer.add_scalar("loss", train_loss, epoch + 1)
            val_bce_writer.add_scalar("loss", val_loss_bce, epoch + 1)
            val_l1_writer.add_scalar("loss", val_loss_l1, epoch + 1)
            iou_writer.add_scalar("iou", iou, epoch + 1)

            if val_loss_l1 < best_val_loss:
                best_val_loss = val_loss_l1
                best_val_loss_epoch = epoch
                iou_at_best_l1 = iou

            if iou > best_iou:
                best_iou = iou
                best_iou_epoch = epoch
                l1_at_best_iou = val_loss_l1

            print("Epoch ", epoch+1, " finished\n")

            if epoch > args.save_epoch:
                torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
                           output_model + "/%02d.pth" % (epoch + 1))

        end_time = datetime.datetime.now()
        print("Finished training")
        print("Least val loss %.4f (iou: %.4f) at epoch %d\n" % (best_val_loss, iou_at_best_l1, best_val_loss_epoch))
        print("Best iou %.4f (l1: %.4f) at epoch %d\n" % (best_iou, l1_at_best_iou, best_iou_epoch))
        print("Time taken: ", start_time.strftime('%D:%H:%M:%S'), " to ", end_time.strftime('%D:%H:%M:%S'))
        train_writer.close()
        val_bce_writer.close()
        val_l1_writer.close()
        iou_writer.close()


def main():
    train_list = []

    for f in sorted(glob.glob(params["shapenet_raytraced"] + args.synset_id + "/*.txt")):
        model_id = f[f.rfind('/') + 1:f.rfind('.')]
        train_list.append({'synset_id': config.synset_id, 'model_id': model_id})

    val_list = train_list[5400:6740]
    train_list = train_list[:5400]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training data size: ", len(train_list))
    print("Validation data size: ", len(val_list))
    print("Device: ", device)

    log_dir = "%s%s/logs/%s/%s_batch%s_trunc%s" % (
        params["network_output"], args.synset_id, args.model_name, args.gt_type, args.batch_size, args.truncation)
    train_writer_path = log_dir + "/train/"
    val_bce_writer_path = log_dir + "/val_bce/"
    val_l1_writer_path = log_dir + "/val_l1/"
    iou_writer_path = log_dir + "/iou/"

    pathlib.Path(train_writer_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_bce_writer_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_l1_writer_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(iou_writer_path).mkdir(parents=True, exist_ok=True)

    train_writer, val_bce_writer, val_l1_writer, iou_writer = create_summary_writers(train_writer_path, val_bce_writer_path, val_l1_writer_path, iou_writer_path)

    trainer = Trainer(train_list, val_list, device)
    trainer.start(train_writer, val_bce_writer, val_l1_writer, iou_writer)


if __name__ == '__main__':
    main()