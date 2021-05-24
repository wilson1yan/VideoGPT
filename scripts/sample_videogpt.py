import argparse
import torch

from videogpt import VideoData, VideoGPT
from videogpt.utils import save_video_grid


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=True)
parser.add_argument('--n', type=int, default=8)
args = parser.parse_args()
n = args.n

gpt = VideoGPT.load_from_checkpoint(args.ckpt).cuda()
gpt.eval()
args = gpt.hparams['args']

args.batch_size = n
data = VideoData(args)
loader = data.test_dataloader()
batch = next(iter(loader))
batch = {k: v.cuda() for k, v in batch.items()}

samples = gpt.sample(n, batch)
save_video_grid(samples, 'samples.mp4')
