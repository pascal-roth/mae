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

# import scripts
import self_sup_seg.third_party.mae.models_mae as models_mae
from self_sup_seg.utils.logger import _logger

# define some local parameters
_COCO_MEAN = np.array([0.485, 0.456, 0.406])  # (IMAGENET) np.array([123.675, 116.280, 103.530])  # (COCO)
_COCO_STD = np.array([0.229, 0.224, 0.225])  # (IMAGENET) np.array([58.395, 57.120, 57.375])  # (COCO)

# necessary to run it on the cluster
matplotlib.use('Agg')


def get_parser():
    parser = argparse.ArgumentParser(description="MAE Visualizer")
    parser.add_argument(
        "--model", "-m",
        default='mae_vit_base_cifar',
        help='model name'
    )
    parser.add_argument(
        "--ckpt",
        default="output_mae_base/checkpoints/last.ckpt",
        help="path to model ckpt",
    )
    # parser.add_argument(
    #     "--input",
    #     nargs="+",
    #     help="A list of space separated input images; "
    #          "or a single glob pattern such as 'directory/*.jpg'",
    # )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--seed",
        default=2,
        help='Random seed to make masking reproducable'
    )
    return parser


def show_image(ax: plt.Axes, image: torch.Tensor, title: str = '') -> None:
    # image is [H, W, 3]
    assert image.shape[2] == 3
    ax.imshow(torch.clip(image * 255, 0, 255).int())
    ax.set_title(title, fontsize=16)
    ax.axis('off')


def prepare_model(path: str, arch: str):
    # get model class
    model_cls = getattr(models_mae, arch)()
    # build model
    if path.endswith('.ckpt'):
        model = model_cls.load_from_checkpoint(path, map_location='cpu')
    elif path.endswith('.pth'):
        model_cls.load_state_dict(torch.load(path, map_location='cpu')['model'], strict=False)
        model = model_cls
    else:
        raise FileNotFoundError(f'Under given path no checkpoint and no model weights found. Path is \n{path}')
    _logger.info(f"Model {type(model).__name__} loaded!")
    _logger.debug(model)
    return model


def run_one_image(img: np.array, model):
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


if __name__ == "__main__":
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)

    # load model
    model_mae = prepare_model(args.ckpt, args.model)

    # load cifar
    import torchvision
    cifar_data = torchvision.datasets.CIFAR10(os.getcwd() + '/sanity_data/', download=True)
    imgs = cifar_data.data[:10]

    # run visualizer
    # if len(args.input) == 1:
    #     args.input = glob.glob(os.path.expanduser(args.input[0]))
    #     assert args.input, "The input path(s) was not found"
    for idx, img in enumerate(tqdm.tqdm(imgs)):
        # use PIL, to be consistent with evaluation
        start_time = time.time()
        print("Image Shape:")
        print(img.shape)
        img = Image.fromarray(img)
        img = img.resize((48, 48))
        img = np.array(img) / 255.
        fig = run_one_image(img, model_mae)
        _logger.info(
            "{}: finished in {:.2f}s".format(
                idx,
                time.time() - start_time,
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, f'{idx}')
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output

            filename, file_extension = os.path.splitext(out_filename)
            if file_extension == '.jpg':
                out_filename = filename + '.png'

            plt.savefig(out_filename, bbox_inches='tight')
        else:
            plt.show()
            assert len(args.input) == 1, "Please specify a directory with args.output, only first output shown now" \
                                         "rest will not be calculated"

