import sys
sys.path.append("../.")
import torch
import losses
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
        self.dataloader_test = torchdata.DataLoader(self.dataset_test)
        # self.dataloader_test = torchdata.DataLoader(self.dataset_test, batch_size=1, shuffle=True,
        #                                             num_workers=2, drop_last=False)

        model = Net(1, 1).to(device)

        # load our saved model and use it to predict the class for test images
        model.load_state_dict(torch.load(saved_model, map_location=self.device))

        mean_iou = 0.0
        counter = 0

        outputs = []

        for idx, sample in enumerate(self.dataloader_test):
            input = sample['occ_grid'].to(self.device)
            target = sample['occ_gt'].to(self.device)

            output = model(input)

            iou_val = metric.iou(output, target)

            mean_iou = mean_iou + iou_val
            counter = counter + 1

            output = output[0].detach()

            outputs.append(output)


            # ===================log========================
            print('[%5d] iou: %.3f' % (idx + 1, iou_val))

        mean_iou = mean_iou / counter

        print("Mean IOU value : ", mean_iou)

        return outputs


if __name__ == '__main__':
    params = JSONHelper.read('../parameters.json')
    # 04468005: 2349848a40065e9f47367565b9fdaec5
    # 04468005: 56687a029d3f46dc52470de2774d6099
    # 04468005: 5588d3f63481259673ad6d3c817cbe81
    # 04468005: 7c511e5744c2ec399d4977b7872dffd3
    # 04468005: 13aac1cbf34edd552470de2774d6099
    # 04468005: 17407a1c24d6a2a58d95cdb16ecced85
    # 04468005: f646c4c40c7f16ae4afcfe0b72eeffb5
    # 04468005: 8823677178c21f28dc14ba0818ee5cec
    # 04468005: 8d136e53fdd5adba8be7c8c5fdb9bd6d
    # 04468005: 691349d753698dd8dc14ba0818ee5cec
    # 04468005: fb26c97b4f84fe5aafe1d4530f4c6e24

    test_list = [{'synset_id': '02747177', 'model_id': 'f53492ed7a071e13cb2a965e75be701c'},
     {'synset_id': '02747177', 'model_id': '5092afb4be0a2f89950ab3eaa7fe7772'},
     {'synset_id': '02747177', 'model_id': '632c8c69e7e7bda54559a6e3650dcd3'},
     {'synset_id': '02747177', 'model_id': 'b689aed9b7017c49f155d75bbf62b80'},
     {'synset_id': '02747177', 'model_id': '4dbbece412ef64b6d2b12aa6a0f050b3'},
     {'synset_id': '02747177', 'model_id': '45d71fc0e59af8bf155d75bbf62b80'},
     {'synset_id': '02747177', 'model_id': 'e19887ca50862a434c86b1fdc94892eb'},
     {'synset_id': '02747177', 'model_id': 'e6ce8ae22ebd6183ad5067eac75a07f7'},
     {'synset_id': '02747177', 'model_id': '5c7bd882d399e031d2b12aa6a0f050b3'},
     {'synset_id': '02747177', 'model_id': 'f249876997cf4fc550da8b99982a3057'},
     {'synset_id': '02747177', 'model_id': '37c9fe32ad87beccad5067eac75a07f7'},
     {'synset_id': '02747177', 'model_id': 'fd013bea1e1ffb27c31c70b1ddc95e3f'},
                 {'synset_id': '04468005', 'model_id': '2349848a40065e9f47367565b9fdaec5'},
                 {'synset_id': '04468005', 'model_id': '56687a029d3f46dc52470de2774d6099'},
                 {'synset_id': '04468005', 'model_id': '5588d3f63481259673ad6d3c817cbe81'},
                 {'synset_id': '04468005', 'model_id': '7c511e5744c2ec399d4977b7872dffd3'},
                 {'synset_id': '04468005', 'model_id': '13aac1cbf34edd552470de2774d6099'}]

    out_folder = params["network_output"]
    saved_model = out_folder + "saved_models/" + config.model_name + ".pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Saved model: ", saved_model)

    tester = Tester(test_list, device)
    outputs = tester.test(saved_model)

    for idx in range(len(test_list)):
        synset_id = test_list[idx]['synset_id']
        model_id = test_list[idx]['model_id']
        output = outputs[idx]
        
        folder = out_folder + synset_id
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

        dataloader.save_sample(folder + "/" + model_id + ".txt", output)
        voxel_grid.txt_to_mesh(folder + "/" + model_id + ".txt", folder + "/" + model_id + ".ply")
        

        

