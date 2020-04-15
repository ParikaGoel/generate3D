import sys
sys.path.append("../.")
import glob
import torch
import pathlib
import argparse
import JSONHelper
from model import *
import data_utils as utils
import eval_metric as metric
import dataset_loader as dataloader
import torch.utils.data as torchdata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = JSONHelper.read('../parameters.json')

# python3 evaluate_model --model_path /home/parika/WorkingDir/complete3D/Assets/output-network/04379243/final_results/Net3D/occ.pth
# --synset_id 04379243 --model_name Net3D --gt_type occ --start_index 6740 --n_vis 30

# command line params
parser = argparse.ArgumentParser()
# model params
parser.add_argument('--model_path', type=str, required=True, help='path to saved model')
parser.add_argument('--synset_id', type=str, required=True, help='synset id of the sample category')
parser.add_argument('--model_name', type=str, required=True, help='which model arch to use')
parser.add_argument('--gt_type', type=str, required=True, help='gt representation')
parser.add_argument('--start_index', type=int, default=6740, help='index to start test set with')
parser.add_argument('--vox_dim', type=int, default=32, help='voxel dim')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--n_vis', type=int, default=20, help='number of visualizations')

args = parser.parse_args()
print(args)

def test(test_list):
    dataset_test = dataloader.DatasetLoad(data_list=test_list, truncation=args.truncation)
    if args.model_name == 'Net3D':
        model = Net3D(1, 1).to(device)
    elif args.model_name == 'UNet3D':
        model = UNet3D(1, 1).to(device)

    dataloader_test = torchdata.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                num_workers=2, drop_last=False)

    # load our saved model and use it to predict the class for test images
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    vis_save = "%s%s/test_vis/%s/%s" % (params["network_output"], args.synset_id, args.model_name,
                                                     args.gt_type)
    pathlib.Path(output_vis).mkdir(parents=True, exist_ok=True)

    n_batches = len(dataloader_test)
    with torch.no_grad():
        for idx, sample in enumerate(dataloader_test):
            input = sample['occ_grid'].to(device)
            names = sample['name']

            if args.gt_type == 'occ':
                target_occ = sample['occ_gt'].to(device)
                target_df = sample['occ_df_gt'].to(device)

                # ===================forward=====================
                output_occ = model(input)

                # Convert occ to df to calculate l1 loss
                output_df = utils.occs_to_dfs(output_occ, trunc=args.truncation, pred=True)
                l1 = losses.l1(output_df, target_df)
                iou = metric.iou_occ(output_occ, target_occ)

                # save the predictions
                if (idx + 1) == n_batches - 1:
                    pred_occs = output_occ[:args.n_vis + 1]
                    target_occs = target_occ[:args.n_vis + 1]
                    names = names[:args.n_vis + 1]
                    utils.save_predictions(vis_save, args.model_name, args.gt_type, names, pred_dfs=None,
                                           target_dfs=None,
                                           pred_occs=pred_occs, target_occs=target_occs)
            else:
                target_df = sample['df_gt'].to(device)

                output_df = model(input)
                l1 = losses.l1(output_df, target_df)
                iou = metric.iou_df(output_df, target_df, trunc_dist=1.0)

                # save the predictions
                if (idx + 1) == n_batches - 1:
                    pred_dfs = output_df[:args.n_vis + 1]
                    target_dfs = target_df[:args.n_vis + 1]
                    names = names[:args.n_vis + 1]
                    utils.save_predictions(vis_save, args.model_name, args.gt_type, names, pred_dfs=pred_dfs,
                                           target_dfs=target_dfs,
                                           pred_occs=None, target_occs=None)

            batch_loss += l1.item()
            batch_iou += iou

        l1_error = batch_loss / (idx + 1)
        mean_iou = batch_iou / (idx + 1)

        print("Mean IOU: ", mean_iou)
        print("L1 Error: ", l1_error)


def main():
    test_list = []

    for f in sorted(glob.glob(params["shapenet_raytraced"] + args.synset_id + "/*.txt")):
        model_id = f[f.rfind('/') + 1:f.rfind('.')]
        test_list.append({'synset_id': args.synset_id, 'model_id': model_id})

    test_list = test_list[args.start_index:]
    test(test_list)


if __name__ == '__main__':
    main()


        

