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
    # synset_test_lst = ['02691156', '02747177', '02773838', '02801938', '02843684', '02933112', '02942699',
    #                    '02946921', '03636649', '03710193', '03759954', '03938244', '04074963', '04099429',
    #                    '04460130', '04468005', '04554684']

    test_list = []

    test_list.append({'synset_id': '03938244', 'model_id': '3fab1dacfa43a7046163a609fcf6c52'})
    test_list.append({'synset_id': '03938244', 'model_id': '4b351af9567719043a4cd082c6787017'})
    test_list.append({'synset_id': '03938244', 'model_id': '4c617e5be0596ee2685998681d42efb8'})
    test_list.append({'synset_id': '03938244', 'model_id': '8b0c10a775c4c4edc1ebca21882cca5d'})
    test_list.append({'synset_id': '03938244', 'model_id': '71dd20123ef5505d5931970d29212910'})

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
        

        

