import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from videogpt import VQVAE, VideoData


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = VQVAE.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='/home/wilson/data/datasets/bair.hdf5')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpus', type=int, default=2)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()
    model = VQVAE(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', mode='min'))

    wandb_logger = WandbLogger(project="moving_mnist_vqvae")

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus)
    trainer = pl.Trainer(callbacks=callbacks, max_steps=20000, **kwargs, logger=wandb_logger)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

