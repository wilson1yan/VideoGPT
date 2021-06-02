import functools
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from .fvd import get_fvd_logits, frechet_distance, load_fvd_model
from videogpt import VideoData, VideoGPT


def main():
    assert torch.cuda.is_available()
    ngpus = torch.cuda.device_count()
    assert FVD_SAMPLE_SIZE % ngpus == 0, f"Must have FVD_SAMPLE_SIZE % n_gpus == 0"

    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), join=True)


def main_worker(rank, size, args_in):
    global args
    args = args_in
    is_root = rank == 0
    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.port}',
                            world_size=size, rank=rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    #################### Load VideoGPT ########################################
    gpt = VideoGPT.load_from_checkpoint(args.ckpt).to(device)
    gpt.eval()
    args = gpt.hparams['args']

    args.batch_size =  16
    loader = VideoData(args).test_dataloader()

    #################### Load I3D ########################################
    i3d = load_fvd_model(device)

    #################### Compute FVD ###############################
    fvds = []
    fvds_star = []
    if is_root:
        pbar = tqdm(total=args.n_trials)
    for _ in range(args.n_trials):
        fvd, fvd_star = eval_fvd(i3d, gpt, test_loader, device)
        fvds.append(fvd)
        fvds_star.append(fvd_star)

        if is_root:
            pbar.update(1)
            fvd_mean = np.mean(fvds)
            fvd_std = np.std(fvds)

            fvd_star_mean = np.mean(fvds_star)
            fvd_star_std = np.std(fvds_star)

            pbar.set_description(f"FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/0 {fvd_star_std:.2f}")
    if is_root:
        pbar.close()
        print(f"Final FVD {fvd_mean:.2f} +/- {fvd_std:.2f}, FVD* {fvd_star_mean:.2f} +/- {fvd_star_std:.2f}")


def all_gather(tensor):
    rank, size = dist.get_rank(), dist.get_world_size()
    tensor_list = [None for _ in range(size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list)


def eval_fvd(sample_fn, i3d, videogpt, loader, device):
    rank, size = dist.get_rank(), dist.get_world_size()
    is_root = rank == 0

    batch = next(iter(loader))

    fake = videogpt.sample(16, batch)
    fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    fake = (fake * 255).astype('uint8')
    fake_embeddings = get_fvd_logits(fake, i3d=i3d, device=device)

    real = batch['video'].to(device)
    real_recon = videogpt.get_reconstruction(real).clamp(0, 1)
    real_recon = real_recon.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real_recon = (real_recon * 255).astype('uint8')
    real_recon_embeddings = get_fvd_logits(real_recon, i3d=i3d, device=device)

    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)

    fake_embeddings = all_gather(fake_embeddings)
    real_recon_embeddings = all_gather(real_recon_embeddings)
    real_embeddings = all_gather(real_embeddings)

    assert fake_embeddings.shape[0] == real_recon_embeddings.shape[0] == real_embeddings.shape[0] == FVD_SAMPLE_SIZE

    if is_root:
        fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
        fvd_star = frechet_distance(fake_embeddings.clone(), real_recon_embeddings)
        return fvd.item(), fvd_star.item()

    return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--n', type=int, default=4, help="Number of trials to compute mean/std")
    parser.add_argument('--port', type=int, default=23452)
    args = parser.parse_args()

    main()
