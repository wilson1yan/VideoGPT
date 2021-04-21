import os
import torch
from torchvision.io import read_video, read_video_timestamps

from videogpt import download, load_vqvae
from videogpt.data import preprocess
import imageio
import gradio as gr
from moviepy.editor import *

device = torch.device('cpu')
vqvae = load_vqvae('kinetics_stride2x4x4', device=device).to(device)

resolution = vqvae.hparams.resolution

def vgpt(invid):
  try:
    os.remove("output.mp4")
  except FileNotFoundError:
    pass
  clip = VideoFileClip(invid)
  rate = clip.fps
  sequence_length=int(clip.fps * clip.duration)
  pts = read_video_timestamps(invid, pts_unit='sec')[0]
  video = read_video(invid, pts_unit='sec', start_pts=pts[0], end_pts=pts[sequence_length - 1])[0]
  video = preprocess(video, resolution, sequence_length).unsqueeze(0).to(device)

  with torch.no_grad():
      encodings = vqvae.encode(video)
      video_recon = vqvae.decode(encodings)
      video_recon = torch.clamp(video_recon, -0.5, 0.5)

  videos = video_recon[0].permute(1, 2, 3, 0) # CTHW -> THWC
  videos = ((videos + 0.5) * 255).cpu().numpy().astype('uint8')
  imageio.mimwrite('output.mp4', videos, fps=int(rate))
  return './output.mp4'

inputs = gr.inputs.Video(label="Input Video")
outputs = gr.outputs.Video(label="Output Video")

title = "VideoGPT"
description = "demo for VideoGPT by University of California, Berkeley. To use it, simply upload your video, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.10157'>VideoGPT: Video Generation using VQ-VAE and Transformers</a> | <a href='https://github.com/wilson1yan/VideoGPT'>Github Repo</a></p>"

examples = [
    ['bear.mp4'],
    ['breakdance.mp4']
]

gr.Interface(vgpt, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(debug=True)