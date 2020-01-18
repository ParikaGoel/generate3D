import argparse
import pickle

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from vocab import Vocabulary

from COCODataloader import get_test_loader, get_loaders
from Triplets.TripletLoss import ContrastiveLoss
from VSEPP_model import VSEPP
from metrics_vsepp import encode_data, i2t, t2i
from utilities.utils import get_glove_100d, check_model_size, get_resnet_model


def createModel(vocab, hidden_dim, device):
    """
    The Triplet Model is created
    :param glove_reduced: glove embedding used
    :param hidden_dim: hidden_dim for the embedder
    :param device: Cuda or Cpu
    :return: Triplet model
    """
    modified_model = get_resnet_model()
    # modified_model.to(device)
    # model = VSEPP(image_embedding_model=modified_model, hidden_dim=hidden_dim, glove=glove_reduced)
    model = VSEPP(image_embedding_model=modified_model, hidden_dim=hidden_dim, vocab=vocab)

    vocab_size = len(vocab)
    model.to(device)
    check_model_size(model)
    return model


def loss_and_optimizer(device, model, lr):
    """
    :param device: Cuda or Cpu
    :param model: The model on which training happens
    :param lr: learning rate to be used
    :return: criterion, optimizer, scheduler
    """
    # criterion = RankingLoss(margin=0.1, gamma=1.5, beta=0.05, k=1, hard_mining=True)
    criterion = ContrastiveLoss(margin=0.2)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=lr, amsgrad=True)
    criterion = criterion.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  # We are trying to minimize loss
    return criterion, optimizer, scheduler, trainable_params


def create_summary_writers(train_writer_path, val_writer_path):
    """
    :param train_writer_path: Path to the train writer
    :param val_writer_path: Path to the val writer
    :return: Summary writer objects
    """
    writer_train = SummaryWriter(train_writer_path)
    writer_val = SummaryWriter(val_writer_path)
    return writer_train, writer_val


def train(model, train_data_loader, optimizer, criterion, epoch, device):
    """
    :param model: TripletRankingNet
    :param train_data_loader:
    :param optimizer: Default is Adam
    :param criterion: TripletRankingLoss
    :param epoch: current epoch
    :param device: cpu or gpu
    :return: train_loss
    """
    batch_loss = 0
    running_loss_iter = 0
    model.start_train()

    for idx, (images, targets, lengths, ids) in enumerate(train_data_loader):
        model.zero_grad()
        images, targets = images.to(device), targets.to(device)
        im, text = model(images, targets, lengths)
        loss = criterion(im, text)
        loss.backward()
        # The gradient clipping is needed for the LSTM section
        clip_grad_norm_(trainable_params, 2.0)
        optimizer.step()
        batch_loss += loss.item()
        running_loss_iter = (running_loss_iter + loss.item())  # just view amt of loss in given iterations
        if idx % 1000 == 0 and idx > 0:  # print on every 1000 iterations
            print("[%d, %5d] loss: %.3f" % (epoch + 1, idx, running_loss_iter / 1000))
            running_loss_iter = 0
            # Also check eval score at this point. Might be handy
            if idx % 1000 == 0:
                validate_score(model, val_loader)
                # Put back to train mode
                model.start_train()
    print("batch loss is {}".format(batch_loss))
    train_loss = batch_loss / (idx + 1)
    return train_loss


def validate(model, val_data_loader, criterion, device):
    """
    :param model: TripletRankingNet that should be used
    :param val_data_loader: validation data loader
    :param criterion: RankingLoss instance
    :param device: cpu or gpu instance
    :return: val_loss
    """
    batch_loss = 0
    model.start_eval()  # Recursively put all its sub-modules to eval
    with torch.no_grad():
        for idx, (images, targets, lengths, ids) in enumerate(val_data_loader):
            images, targets = images.to(device), targets.to(device)
            im, text = model(images, targets, lengths)
            loss = criterion(im, text)
            batch_loss += loss.item()
        print("batch loss is %.3f" % (batch_loss))
        val_loss = batch_loss / (idx + 1)
        return val_loss


def validate_score(model, val_loader):
    model.start_eval()
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(model, val_loader)
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs)
    print(f"Image to text: {r1}, {r5}, {r10}, {medr}, {meanr})")
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs)
    print(f"Text to image: {r1i}, {r5i}, {r10i}, {medri}, {meanr})")
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    return currscore


def adjust_learning_rate(learning_rate, optimizer, epoch):
    if epoch == 15:
        learning_rate = learning_rate / 10
        print(f"Updating the learning rate to {learning_rate}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate


def main(num_epoch, train_data_loader, val_data_loader, optimizer, writer_train, writer_val, device, batch_size,
         criterion):
    print("Execution started")
    max_recall_sum = 0
    for epoch in range(num_epoch):
        # Check if the learning rate needs to be adjusted
        adjust_learning_rate(args.lr, optimizer, epoch)
        train_loss = train(model, train_data_loader, optimizer=optimizer, criterion=criterion, epoch=epoch,
                           device=device)
        val_loss = validate(model, val_data_loader, criterion=criterion, device=device)  # , val_acc
        writer_train.add_scalar("loss", train_loss, epoch + 1)
        writer_val.add_scalar("loss", val_loss, epoch + 1)
        recall_sum = validate_score(model, val_data_loader)
        writer_val.add_scalar("recall sum", recall_sum, epoch + 1)
        if recall_sum > max_recall_sum:
            max_recall_sum = recall_sum
            torch.save(model.state_dict(), model_save_path + model_name.format(epoch))
            torch.save(optimizer.state_dict(), model_save_path + optim_name.format(epoch))
        print("epoch {} complete".format(epoch))
        # scheduler.step(train_loss)
    writer_train.close()
    writer_val.close()


def create_dataloaders(vocab, args):
    # Transforms to be used
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          # transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),
                                          ])
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                        ])
    train_loader, val_loader = get_loaders(
        dpath="/home/chinmay/Desktop/Dataset/data/coco", vocab=vocab, train_transform=train_transform,
        val_transform=val_transform, batch_size=args.batch_size, num_workers=12
    )
    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running Classifier Network')
    parser.add_argument("-epoch", default=30, type=int, help="Number of epochs to run")
    parser.add_argument("-batch_size", default=128, type=int, help="Batch size to be used for training")
    parser.add_argument("-hidden_dim", type=int, default=1024, help="Size of hidden dimensions of the network")
    parser.add_argument("-counter", type=int, default=5, help="The name to append to writers")
    parser.add_argument("-lr", type=int, default=2 * 1e-4, help="The learning rate used")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "/home/chinmay/Desktop/saved_models/vsepp/"
    model_name = "vsepp-{}.pth"
    optim_name = "vsepp-optim-{}.pth"

    vocab = pickle.load(open('/home/chinmay/Desktop/coco-updated/vocab/coco_vocab.pkl', 'rb'))
    # data_name, vocab, batch_size, train_transform, val_transform, num_workers

    train_loader, val_loader = create_dataloaders(vocab, args)
    model = createModel(vocab, args.hidden_dim, device)
    criterion, optimizer, scheduler, trainable_params = loss_and_optimizer(device, model, args.lr)
    train_writer_path = "/home/chinmay/Desktop/logs/vsepp/train/" + str(args.counter)
    val_writer_path = "/home/chinmay/Desktop/logs/vsepp/val/" + str(args.counter)
    train_writer, val_writer = create_summary_writers(train_writer_path, val_writer_path)

    main(num_epoch=args.epoch, train_data_loader=train_loader, val_data_loader=val_loader, optimizer=optimizer,
         writer_train=train_writer, writer_val=val_writer, device=device, batch_size=args.batch_size,
         criterion=criterion)