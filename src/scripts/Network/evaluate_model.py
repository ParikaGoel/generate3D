import sys
sys.path.append("../.")
import glob
import torch
import config
import pathlib
import JSONHelper
import voxel_grid
from model import *
import eval_metric as metric
import dataset_loader as dataloader
import torch.utils.data as torchdata


class Tester:
    def __init__(self, test_list, device):
        self.dataset_test = dataloader.DatasetLoad(test_list)
        self.device = device

    def test(self, saved_model):
        self.dataloader_test = torchdata.DataLoader(self.dataset_test, batch_size=1, shuffle=True,
                                                    num_workers=2, drop_last=False)

        model = Net(1, 1).to(device)

        # load our saved model and use it to predict the class for test images
        model.load_state_dict(torch.load(saved_model, map_location=self.device))

        mean_iou = 0.0
        largest_iou = 0.0
        counter = 0

        outputs = []

        for idx, sample in enumerate(self.dataloader_test):
            input = sample['occ_grid'].to(self.device)
            target = sample['occ_gt'].to(self.device)
            output = model(input)

            iou_val = metric.iou(torch.nn.Sigmoid()(output), target)

            mean_iou = mean_iou + iou_val

            if (largest_iou < iou_val):
                largest_iou = iou_val
                best_predicted_model = output[0].detach()

            counter = counter + 1

            outputs.append(output[0].detach())


            # ===================log========================
            print('[%5d] iou: %.3f' % (idx + 1, iou_val))

        mean_iou = mean_iou / counter

        print("Mean IOU value : ", mean_iou)
        print("Largest IOU value : ", largest_iou)

        return outputs, best_predicted_model


if __name__ == '__main__':
    params = JSONHelper.read('../parameters.json')

    synset_test_lst = ['04379243']
    test_list = []

    for synset_id in synset_test_lst:
        for f in glob.glob(params["shapenet_raytraced"] + synset_id + "/*.txt"):
            model_id = f[f.rfind('/') + 1:f.rfind('.')]
            test_list.append({'synset_id': synset_id, 'model_id': model_id})

    test_list = test_list[6740:]

    out_folder = params["network_output"]
    saved_model = out_folder + "saved_models/" + config.model_name + ".pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Models being tested: ", test_list)
    print("Saved model: ", saved_model)

    tester = Tester(test_list, device)
    outputs, best_model = tester.test(saved_model)

    for idx in range(len(test_list)):
        synset_id = test_list[idx]['synset_id']
        model_id = test_list[idx]['model_id']
        output = outputs[idx]

        folder = out_folder + "predicted_test_output/" + config.model_name + "/" + synset_id
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

        dataloader.save_sample(folder + "/" + model_id + ".txt", output)
        voxel_grid.txt_to_mesh(folder + "/" + model_id + ".txt", folder + "/" + model_id + ".ply")

    dataloader.save_sample(folder + "/best_model.txt", best_model)
    voxel_grid.txt_to_mesh(folder + "/best_model.txt", folder + "/best_model.ply")
        

        

