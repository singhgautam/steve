import argparse
import numpy as np

import torch

from torch.utils.data import DataLoader

from steve import STEVE
from data import GlobVideoDatasetWithMasks
from ari import evaluate_ari


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--ep_len', type=int, default=6)

parser.add_argument('--trained_model_paths', nargs='+', default=[
    'logs_for_seed_1/best_model_until_200000_steps.pt',
    'logs_for_seed_2/best_model_until_200000_steps.pt',
    'logs_for_seed_3/best_model_until_200000_steps.pt',
])
parser.add_argument('--data_path', default='eval_data/*')
parser.add_argument('--data_num_segs_per_frame', type=int, default=25)

parser.add_argument('--num_iterations', type=int, default=2)
parser.add_argument('--num_slots', type=int, default=15)
parser.add_argument('--cnn_hidden_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_predictor_blocks', type=int, default=1)
parser.add_argument('--num_predictor_heads', type=int, default=4)
parser.add_argument('--predictor_dropout', type=int, default=0.0)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_blocks', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

args = parser.parse_args()

torch.manual_seed(args.seed)

eval_dataset = GlobVideoDatasetWithMasks(root=args.data_path, phase='full', img_size=args.image_size, num_segs=args.data_num_segs_per_frame,
                                        ep_len=args.ep_len, img_glob='????????_image.png')

eval_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, **loader_kwargs)

models = []
for path in args.trained_model_paths:
    model = STEVE(args)
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.cuda()
    models += [model]

with torch.no_grad():
    for model in models:
        model.eval()

    fgaris = []
    for batch, (video, true_masks) in enumerate(eval_loader):
        video = video.cuda()

        fgaris_b = []
        for model in models:
            _, _, pred_masks_b_m = model.encode(video)

            # omit the BG segment i.e. the 0-th segment from the true masks as follows.
            fgari_b_m = 100 * evaluate_ari(true_masks.permute(0, 2, 1, 3, 4, 5)[:, 1:].flatten(start_dim=2),
                                           pred_masks_b_m.permute(0, 2, 1, 3, 4, 5).flatten(start_dim=2))
            fgaris_b += [fgari_b_m]

        fgaris += [fgaris_b]

        # print results
        fgaris_numpy = np.asarray(fgaris)
        mean = fgaris_numpy.mean(axis=0).mean()
        stddev = fgaris_numpy.mean(axis=0).std()
        print(f"Done batches {batch + 1}. Over {len(models)} seeds, \t FG-ARI MEAN = {mean:.3f} \t STD = {stddev:.3f} .")
