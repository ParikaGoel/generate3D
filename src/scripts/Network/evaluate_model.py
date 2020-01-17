import sys
sys.path.append("../.")
import torch
import pathlib
import voxel_grid
from model import *
import eval_metric as metric
import dataset_loader as dataloader
from torch.autograd import Variable
import torch.utils.data as torchdata


class Tester:
    def __init__(self, test_list):
        self.dataset_test = dataloader.DatasetLoad(test_list)

    def test(self):
        self.dataloader_test = torchdata.DataLoader(self.dataset_test)
        # self.dataloader_test = torchdata.DataLoader(self.dataset_test, batch_size=1, shuffle=True,
        #                                             num_workers=2, drop_last=False)

        model = Net(1, 1).cpu()

        # load our saved model and use it to predict the class for test images
        model.load_state_dict(torch.load('/home/parika/WorkingDir/complete3D/Assets/output-network/model.pth'))

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
    test_list = []
    # test_list.append({'synset_id': '02747177', 'model_id': '9ee464a3fb9d3e8e57cd6640bbeb736d'})
    # test_list.append({'synset_id': '02747177', 'model_id': 'f67bf3e49cfffc91f155d75bbf62b80'})
    # test_list.append({'synset_id': '02747177', 'model_id': '60b4017e16fd9c4f2056b4bd5d870b47'})
    # test_list.append({'synset_id': '02747177', 'model_id': '2f79bca58f58a3ead2b12aa6a0f050b3'})
    # test_list.append({'synset_id': '02747177', 'model_id': 'af1dc226f465990e81faebbdea6bd9be'})

    test_list.append({'synset_id': '02773838', 'model_id': '4e4fcfffec161ecaed13f430b2941481'})
    test_list.append({'synset_id': '02773838', 'model_id': '3077a9b76724b6d35de21284bb515a83'})
    test_list.append({'synset_id': '02773838', 'model_id': 'cbc2328cadf8dc573394926146371698'})
    test_list.append({'synset_id': '02773838', 'model_id': 'd3bd250ca3cb8e29976855a35549333'})
    test_list.append({'synset_id': '02773838', 'model_id': '7565e6f425dd6d376d987ae9a225629c'})

    # test_list.append({'synset_id': '02933112', 'model_id': '2f0fd2a5e181b82a4267f85fb94fa2e7'})
    # test_list.append({'synset_id': '02933112', 'model_id': 'a46d947577ecb54a6bdcd672c2b17215'})
    # test_list.append({'synset_id': '02933112', 'model_id': '37ba0371250bcd6de117ecc943aca233'})
    # test_list.append({'synset_id': '02933112', 'model_id': 'bd2bcee265b1ee1c7c373e0e7470a338'})
    # test_list.append({'synset_id': '02933112', 'model_id': '8a2aadf8fc4f092c5ee1a94f1da3a5e'})

    tester = Tester(test_list)
    outputs = tester.test()
    
    out_folder = "/home/parika/WorkingDir/complete3D/Assets/output-network/"

    for idx in range(len(test_list)):
        synset_id = test_list[idx]['synset_id']
        model_id = test_list[idx]['model_id']
        output = outputs[idx]
        
        folder = out_folder + synset_id
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

        dataloader.save_sample(folder + "/" + model_id + ".txt", output)
        voxel_grid.txt_to_mesh(folder + "/" + model_id + ".txt", folder + "/" + model_id + ".ply")
        

        

