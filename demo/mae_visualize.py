# isualization script for MAE
# Author: Pascal Roth

# import packages
import os
import argparse
import glob
import tqdm
import time
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import requests

# import scripts
import self_sup_seg.third_party.mae.models_mae as models_mae
import self_sup_seg.third_party.mae.models_vicreg as models_vicreg
import self_sup_seg.third_party.mae.models_swin as models_swin
from self_sup_seg.third_party.mae.params import IMAGE_MEAN, IMAGE_STD
from self_sup_seg.utils.logger import _logger

# necessary to run it on the cluster
matplotlib.use('Agg')


def get_parser():
    parser = argparse.ArgumentParser(description="MAE Visualizer")
    parser.add_argument(
        "--model", "-m",
        default='mae_vit_base_patch16',
        help='model name'
    )
    parser.add_argument(
        "--path",
        default="output_mae_img_loss/checkpoints/last.ckpt",
        help="path to model ckpt or weight pth file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default = None,
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'."
             "If not given, will take example image",
    )
    parser.add_argument(
        "--output",
        default="output_mae_img_loss",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--seed",
        default=2,
        help='Random seed to make masking reproducable'
    )
    parser.add_argument(
        '--swin', 
        action='store_true',
        help='decide if swin architecture should be used'
    )
    parser.add_argument(
        '--vic', 
        action='store_true',
        help='Activate VicReg Loss (only for ViT Models)'
    )   
    return parser


def read_image(path: str):
    if path:
        assert os.path.isfile(path), f'Given path does not lead to image. Path is "{path}"'
        img = Image.open(path)
    else:
        img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
        # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
        img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - IMAGE_MEAN
    img = img / IMAGE_STD
    return img


def show_image(ax: plt.Axes, image: torch.Tensor, title: str = '') -> None:
    # image is [H, W, 3]
    assert image.shape[2] == 3
    ax.imshow(torch.clip((image * np.array(IMAGE_STD) + np.array(IMAGE_MEAN)) * 255, 0, 255).int())
    ax.set_title(title, fontsize=16)
    ax.axis('off')


def prepare_model(path: str, arch: str, swin: bool = False, vic: bool = False):
    # get model class
    if swin:
        model_cls = getattr(models_swin, arch)() 
    elif vic:
        model_cls = getattr(models_vicreg, arch)()
    else:
        model_cls = getattr(models_mae, arch)()

    # build model
    if path.endswith('.ckpt'):
        model = model_cls.load_from_checkpoint(path, map_location='cpu')
    elif path.endswith('.pth'):
        msg = model_cls.load_state_dict(torch.load(path, map_location='cpu')['model'], strict=False)
        print(msg)
        model = model_cls
    else:
        raise FileNotFoundError(f'Under given path no checkpoint and no model weights found. Path is \n{path}')
    _logger.info(f"Model {type(model).__name__} loaded!")
    _logger.debug(model)
    return model


def run_one_image(img: np.array, model, swin):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    # Einsum explanation: Sums the product of the elements of the input along dimensions specified using a notation
    # based on the Einstein summation convention.

    # run MAE
    with torch.no_grad():
        y, mask = model(x.float())
        loss = model.forward_loss(x.float(), y, mask)
        if not swin:
            y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 6]
    fig, axs = plt.subplots(1, 4)

    fig.suptitle(f'Image Visualization (loss {np.round(loss, decimals=6)})')
    show_image(axs[0], x[0], "original")
    show_image(axs[1], im_masked[0], "masked")
    show_image(axs[2], y[0], "reconstruction")
    show_image(axs[3], im_paste[0], "reconstruction + visible")

    plt.tight_layout()
    return fig


def run_visualizer(args, model, name_extension: str = None):
    # run visualizer
    if args.input and len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    elif not args.input: 
        args.input = [None]

    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path)
        start_time = time.time()
        print("Image Shape:")
        print(img.shape)
        fig = run_one_image(img, model, args.swin)
        _logger.info(
            "{}: finished in {:.2f}s".format(
                path if path else 'Example',
                time.time() - start_time,
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path)) if path else os.path.join(args.output, 'example_image.png')
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output

            filename, file_extension = os.path.splitext(out_filename)
            if file_extension == '.jpg':
                out_filename = filename + '.png'

            if name_extension:
                out_filename = filename + name_extension + '.png'

            plt.savefig(out_filename, bbox_inches='tight')
        else:
            plt.show()
            assert len(args.input) == 1, "Please specify a directory with args.output, only first output shown now" \
                                         "rest will not be calculated"
        plt.close()


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    torch.manual_seed(args.seed)

    # load model
    if True:
        model = prepare_model(args.path, args.model, args.swin, args.vic)
        run_visualizer(args, model)

    else:
        torch.manual_seed(args.seed)
        args.input = None
        model_full = prepare_model('/media/pascal/TOSHIBA EXT/ETH/mae_pretrain_vit_large_full.pth', args.model)
        run_visualizer(args, model_full, name_extension='_full')
        
        torch.manual_seed(args.seed)
        args.input = None
        model_demo = prepare_model('/media/pascal/TOSHIBA EXT/ETH/mae_visualize_vit_large.pth', args.model)  
        run_visualizer(args, model_demo, name_extension='_demo')
        
        torch.manual_seed(args.seed)
        args.input = None
        model_fine = prepare_model('/media/pascal/TOSHIBA EXT/ETH/mae_finetuned_vit_large.pth', args.model)  # decoder weights are missing
        run_visualizer(args, model_fine, name_extension='_fine')

        torch.manual_seed(args.seed)
        args.input=None
        model_gan  = prepare_model('/media/pascal/TOSHIBA EXT/ETH/mae_visualize_vit_large_ganloss.pth', args.model)
        run_visualizer(args, model_gan,  name_extension='_gan')

        torch.manual_seed(args.seed)
        args.input = None
        model_mae  = prepare_model('/home/pascal/ETH/2_semester/project/mae_pretrain_vit_large.pth', args.model)  # decoder weights are missing
        run_visualizer(args, model_mae,  name_extension='_mae')
