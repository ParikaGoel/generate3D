import sys
sys.path.append("../.")
import torch
import config
import pathlib
import JSONHelper
import voxel_grid
from model import *
import eval_metric as metric
import dataset_loader as dataloader
from torch.autograd import Variable
import torch.utils.data as torchdata


class Tester:
    def __init__(self, test_list):
        self.dataset_test = dataloader.DatasetLoad(test_list)

    def test(self, saved_model):
        self.dataloader_test = torchdata.DataLoader(self.dataset_test)
        # self.dataloader_test = torchdata.DataLoader(self.dataset_test, batch_size=1, shuffle=True,
        #                                             num_workers=2, drop_last=False)

        model = Net(1, 1).cpu()

        # load our saved model and use it to predict the class for test images
        model.load_state_dict(torch.load(saved_model))

        mean_iou = 0.0
        counter = 0

        outputs = []

        for idx, sample in enumerate(self.dataloader_test):
            input = Variable(sample['occ_grid']).cpu()
            target = Variable(sample['occ_gt']).cpu()

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
    # synset_test_lst = ['02691156', '02747177', '02773838', '02801938', '02843684', '02933112', '02942699',
    #                    '02946921', '03636649', '03710193', '03759954', '03938244', '04074963', '04099429',
    #                    '04460130', '04468005', '04554684']

    # test_list = []
    #
    # test_list.append({'synset_id': '03938244', 'model_id': '3fab1dacfa43a7046163a609fcf6c52'})
    # test_list.append({'synset_id': '03938244', 'model_id': '4b351af9567719043a4cd082c6787017'})
    # test_list.append({'synset_id': '03938244', 'model_id': '4c617e5be0596ee2685998681d42efb8'})
    # test_list.append({'synset_id': '03938244', 'model_id': '8b0c10a775c4c4edc1ebca21882cca5d'})
    # test_list.append({'synset_id': '03938244', 'model_id': '71dd20123ef5505d5931970d29212910'})

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
     {'synset_id': '02747177', 'model_id': 'fd013bea1e1ffb27c31c70b1ddc95e3f'}]

    # test_list = [{'synset_id': '02747177', 'model_id': 'fd013bea1e1ffb27c31c70b1ddc95e3f'}]

    out_folder = params["network_output"]
    saved_model = out_folder + "saved_models/" + config.model_name + ".pth"

    print("Saved model: ", saved_model)

    tester = Tester(test_list)
    outputs = tester.test(saved_model)

    for idx in range(len(test_list)):
        synset_id = test_list[idx]['synset_id']
        model_id = test_list[idx]['model_id']
        output = outputs[idx]
        
        folder = out_folder + synset_id
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

        dataloader.save_sample(folder + "/" + model_id + ".txt", output)
        voxel_grid.txt_to_mesh(folder + "/" + model_id + ".txt", folder + "/" + model_id + ".ply")
        

        

