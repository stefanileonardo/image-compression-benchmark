# ---------------------------------------------------------------------
# Modified train_model.py for SMoE on Kaggle
# - TF 1.x compat mode
# - Saves plain reconstruction.png after training
# ---------------------------------------------------------------------

# for execution without a display
# import matplotlib as mpl
# mpl.use('Agg')

import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import shutil
from skimage.io import imsave

# --- TensorFlow 1.x compatibility ---
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1

from smoe import Smoe
from plotter import ImagePlotter, LossPlotter
from logger import ModelLogger
from utils import save_model, load_params


def main(
    image_path,
    results_path,
    iterations,
    validation_iterations,
    kernels_per_dim,
    params_file,
    l1reg,
    base_lr,
    batches,
    checkpoint_path,
    lr_div,
    lr_mult,
    disable_train_pis,
    disable_train_gammas,
    radial_as,
    quiet,
):
    # --- load image ---
    orig = plt.imread(image_path)
    if orig.dtype == np.uint8:
        orig = orig.astype(np.float32) / 255.0

    # --- init params ---
    if params_file is not None:
        init_params = load_params(params_file)
    else:
        init_params = None

    # --- results dir ---
    if results_path is not None:
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path, exist_ok=True)

    # --- plotters + logger ---
    loss_plotter = LossPlotter(path=results_path + "/loss.png", quiet=quiet)
    image_plotter = ImagePlotter(
        path=results_path,
        options=["orig", "reconstruction", "gating", "pis_hist"],
        quiet=quiet,
    )
    logger = ModelLogger(path=results_path)

    # --- SMoE model ---
    smoe = Smoe(
        orig,
        kernels_per_dim,
        init_params=init_params,
        train_pis=not disable_train_pis,
        train_gammas=not disable_train_gammas,
        radial_as=radial_as,
        start_batches=batches,
    )

    # --- optimizers ---
    optimizer1 = tf.train.AdamOptimizer(base_lr)
    optimizer2 = tf.train.AdamOptimizer(base_lr / lr_div)
    optimizer3 = tf.train.AdamOptimizer(base_lr * lr_mult)

    # optimizers have to be set before restore
    smoe.set_optimizer(optimizer1, optimizer2, optimizer3)

    # --- optional checkpoint restore ---
    if checkpoint_path is not None:
        smoe.restore(checkpoint_path)

    # --- training ---
    smoe.train(
        iterations,
        val_iter=validation_iterations,
        pis_l1=l1reg,
        callbacks=[loss_plotter.plot, image_plotter.plot, logger.log],
    )

    # --- save plain reconstruction image for tiling code ---
    recon = smoe.get_reconstruction()  # float array in [0,1]
    recon_uint8 = (recon * 255.0).clip(0, 255).astype(np.uint8)
    recon_path = os.path.join(results_path, "reconstruction.png")
    imsave(recon_path, recon_uint8)
    print("Saved reconstruction image to", recon_path)

    # --- save parameters ---
    save_model(smoe, results_path + "/params_best.pkl", best=True)
    save_model(smoe, results_path + "/params_last.pkl", best=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image_path", type=str, required=True, help="input image"
    )
    parser.add_argument(
        "-r", "--results_path", type=str, required=True, help="results path"
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=100,
        help="number of iterations",
    )
    parser.add_argument(
        "-v",
        "--validation_iterations",
        type=int,
        default=100,
        help="number of iterations between validations",
    )
    parser.add_argument(
        "-k",
        "--kernels_per_dim",
        type=int,
        default=12,
        help="number of kernels per dimension",
    )
    parser.add_argument(
        "-p",
        "--params_file",
        type=str,
        default=None,
        help="parameter file for model initialization",
    )
    parser.add_argument(
        "-reg",
        "--l1reg",
        type=float,
        default=0.0,
        help="l1 regularization for pis",
    )
    parser.add_argument(
        "-lr",
        "--base_lr",
        type=float,
        default=0.001,
        help="base learning rate",
    )
    parser.add_argument(
        "-b",
        "--batches",
        type=int,
        default=1,
        help=(
            "number of batches to split the training into "
            "(will be automatically reduced when number of pis drops)"
        ),
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "path to a checkpoint file to continue the training. "
            "EXPERIMENTAL."
        ),
    )
    parser.add_argument(
        "-d", "--lr_div", type=float, default=100.0, help="div for pis lr"
    )
    parser.add_argument(
        "-m", "--lr_mult", type=float, default=1000.0, help="mult for a lr"
    )
    parser.add_argument(
        "-dp",
        "--disable_train_pis",
        type=bool,
        default=False,
        help="disable training of pis",
    )
    parser.add_argument(
        "-dg",
        "--disable_train_gammas",
        type=bool,
        default=False,
        help="disable training of gammas",
    )
    parser.add_argument(
        "-ra",
        "--radial_as",
        type=bool,
        default=False,
        help="use radial kernel (no steering)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        type=bool,
        default=False,
        help="do not display plots",
    )

    args = parser.parse_args()
    main(**vars(args))
