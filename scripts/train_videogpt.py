import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from videogpt import VideoGPT, VideoData


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()

    # Add args that originally came from this line below:
    # parser = pl.Trainer.add_argparse_args(parser)
    # ...but don't work anymore in the new version of PyTorch Lightning
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--amp_level', type=str, default='')
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=20*1000)

    parser = VideoGPT.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    args.class_cond_dim = data.n_classes if args.class_cond else None
    model = VideoGPT(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=-1))

    wandb_logger = WandbLogger(project="moving_mnist_videogpt")

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(devices=args.gpus, accelerator="gpu", # gpus=args.gpus,
                      strategy='ddp')
    trainer = pl.Trainer(callbacks=callbacks, max_steps=args.max_steps, **kwargs, logger=wandb_logger)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

