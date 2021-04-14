import requests
from tqdm import tqdm
import os
import torch

from .vqvae import VQVAE


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 8192

    pbar = tqdm(total=0, unit='iB', unit_scale=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()


def download(id, fname, root=os.path.expanduser('~/.cache/videogpt')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    URL = 'https://drive.google.com/uc?export=download'
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
    return destination


_VQVAE = {
    'bair_stride4x2x2': '1iIAYJ2Qqrx5Q94s5eIXQYJgAydzvT_8L', # trained on 16 frames of 64 x 64 images
    'ucf101_stride4x4x4': '1uuB_8WzHP_bbBmfuaIV7PK_Itl3DyHY5', # trained on 16 frames of 128 x 128 images
    'kinetics_stride4x4x4': '1DOvOZnFAIQmux6hG7pN_HkyJZy3lXbCB', # trained on 16 frames of 128 x 128 images
    'kinetics_stride2x4x4': '1jvtjjtrtE4cy6pl7DK_zWFEPY3RZt2pB' # trained on 16 frames of 128 x 128 images
}

def load_vqvae(model_name, device=torch.device('cpu')):
    assert model_name in _VQVAE, f"Invalid model_name: {model_name}"
    filepath = download(_VQVAE[model_name], model_name)
    vqvae = VQVAE.load_from_checkpoint(filepath).to(device)
    vqvae.eval()

    return vqvae
