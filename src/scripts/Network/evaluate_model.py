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

synset_id = '04379243'

class Tester:
    def __init__(self, test_list, device):
        self.dataset_test = dataloader.DatasetLoad(test_list)
        self.device = device

    def test(self, saved_model):
        self.dataloader_test = torchdata.DataLoader(self.dataset_test, batch_size=1, shuffle=True,
                                                    num_workers=2, drop_last=False)

        model = Net3(1, 1)

        # load our saved model and use it to predict the class for test images
        model.load_state_dict(torch.load(saved_model, map_location=self.device))
        model = model.to(device)
        model.eval()

        mean_iou = 0.0
        largest_iou = 0.0
        counter = 0

        with torch.no_grad():
            for idx, sample in enumerate(self.dataloader_test):
                input = sample['occ_grid'].to(self.device)
                target = sample['df_gt'].to(self.device)

                output = model(input)

                iou_val = metric.iou_df(output, target, 1.0)

                mean_iou = mean_iou + iou_val

                if (largest_iou < iou_val):
                    largest_iou = iou_val
                    best_predicted_model = output[0].detach()
                    ground_truth = target[0].detach()

                counter = counter + 1


                # ===================log========================
                print('[%5d] iou: %.3f' % (idx + 1, iou_val))

            mean_iou = mean_iou / counter

            print("Mean IOU value : ", mean_iou)
            print("Largest IOU value: ", largest_iou)

        return best_predicted_model, ground_truth


if __name__ == '__main__':
    params = JSONHelper.read('../parameters.json')

    test_list = []

    for f in glob.glob(params["shapenet_raytraced"] + synset_id + "/*.txt"):
        model_id = f[f.rfind('/') + 1:f.rfind('.')]
        test_list.append({'synset_id': synset_id, 'model_id': model_id})

    test_list = test_list[6740:]

    out_folder = params["network_output"] + 'Net3/' + synset_id
    saved_model = out_folder + "/saved_models/df.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Saved model: ", saved_model)

    tester = Tester(test_list, device)
    best_model, ground_truth = tester.test(saved_model)

    dataloader.save_df(out_folder + "/predicted_test_output/best_model", best_model)
    # loader.df_to_mesh(out_folder + "/predicted_test_output/best_model.npy", out_folder + "/predicted_test_output/best_model.ply")

    dataloader.save_df(out_folder + "/predicted_test_output/gt", ground_truth)
    # loader.df_to_mesh(out_folder + "/predicted_test_output/gt.npy",
    #                        out_folder + "/predicted_test_output/gt.ply")
        

        

