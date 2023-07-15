import argparse

HPARAMS_REGISTRY = {}


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


morphomnist = Hparams()
morphomnist.lr = 1e-3
morphomnist.bs = 32
morphomnist.wd = 0.01
morphomnist.z_dim = 16
morphomnist.input_res = 32
morphomnist.pad = 4
morphomnist.enc_arch = "32b3d2,16b3d2,8b3d2,4b3d4,1b4"
morphomnist.dec_arch = "1b4,4b4,8b4,16b4,32b4"
morphomnist.widths = [16, 32, 64, 128, 256]
morphomnist.parents_x = ["thickness", "intensity", "digit"]
morphomnist.concat_pa = True
morphomnist.context_norm = "[-1,1]"
morphomnist.context_dim = 12
HPARAMS_REGISTRY["morphomnist"] = morphomnist


cmnist = Hparams()
cmnist.lr = 1e-3
cmnist.bs = 32
cmnist.wd = 0.01
cmnist.z_dim = 16
cmnist.input_res = 32
cmnist.input_channels = 3
cmnist.pad = 4
cmnist.enc_arch = "32b3d2,16b3d2,8b3d2,4b3d4,1b4"
cmnist.dec_arch = "1b4,4b4,8b4,16b4,32b4"
cmnist.widths = [16, 32, 64, 128, 256]
cmnist.parents_x = ["digit", "colour"]
cmnist.context_dim = 20
HPARAMS_REGISTRY["cmnist"] = cmnist


ukbb64 = Hparams()
ukbb64.lr = 1e-3
ukbb64.bs = 32
ukbb64.wd = 0.1
ukbb64.z_dim = 16
ukbb64.input_res = 64
ukbb64.pad = 3
ukbb64.enc_arch = "64b3d2,32b31d2,16b15d2,8b7d2,4b3d4,1b2"
ukbb64.dec_arch = "1b2,4b4,8b8,16b16,32b32,64b4"
ukbb64.widths = [32, 64, 128, 256, 512, 1024]
HPARAMS_REGISTRY["ukbb64"] = ukbb64


ukbb192 = Hparams()
ukbb192.update(ukbb64.__dict__)
ukbb192.input_res = 192
ukbb192.pad = 9
ukbb192.enc_arch = "192b1d2,96b3d2,48b7d2,24b11d2,12b7d2,6b3d6,1b2"
ukbb192.dec_arch = "1b2,6b4,12b8,24b12,48b8,96b4,192b2"
ukbb192.widths = [32, 64, 96, 128, 160, 192, 512]
HPARAMS_REGISTRY["ukbb192"] = ukbb192


mimic192 = Hparams()
mimic192.lr = 1e-3
mimic192.bs = 16
mimic192.wd = 0.1
mimic192.z_dim = 16
mimic192.input_res = 192
mimic192.pad = 9
mimic192.enc_arch = "192b1d2,96b3d2,48b7d2,24b11d2,12b7d2,6b3d6,1b2"
mimic192.dec_arch = "1b2,6b4,12b8,24b12,48b8,96b4,192b2"
mimic192.widths = [32, 64, 96, 128, 160, 192, 512]
HPARAMS_REGISTRY["mimic192"] = mimic192


def setup_hparams(parser: argparse.ArgumentParser) -> Hparams:
    hparams = Hparams()
    args = parser.parse_known_args()[0]
    valid_args = set(args.__dict__.keys())
    hparams_dict = HPARAMS_REGISTRY[args.hps].__dict__
    for k in hparams_dict.keys():
        if k not in valid_args:
            raise ValueError(f"{k} not in default args")
    parser.set_defaults(**hparams_dict)
    hparams.update(parser.parse_known_args()[0].__dict__)
    return hparams


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument(
        "--data_dir", help="Data directory to load form.", type=str, default=""
    )
    parser.add_argument("--hps", help="hyperparam set.", type=str, default="ukbb64")
    parser.add_argument(
        "--resume", help="Path to load checkpoint.", type=str, default=""
    )
    parser.add_argument("--seed", help="Set random seed.", type=int, default=7)
    parser.add_argument(
        "--deterministic",
        help="Toggle cudNN determinism.",
        action="store_true",
        default=False,
    )
    # training
    parser.add_argument("--epochs", help="Training epochs.", type=int, default=5000)
    parser.add_argument("--bs", help="Batch size.", type=int, default=32)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-3)
    parser.add_argument(
        "--lr_warmup_steps", help="lr warmup steps.", type=int, default=100
    )
    parser.add_argument("--wd", help="Weight decay penalty.", type=float, default=0.01)
    parser.add_argument(
        "--betas",
        help="Adam beta parameters.",
        nargs="+",
        type=float,
        default=[0.9, 0.9],
    )
    parser.add_argument(
        "--ema_rate", help="Exp. moving avg. model rate.", type=float, default=0.999
    )
    parser.add_argument(
        "--input_res", help="Input image crop resolution.", type=int, default=64
    )
    parser.add_argument(
        "--input_channels", help="Input image num channels.", type=int, default=1
    )
    parser.add_argument("--pad", help="Input padding.", type=int, default=3)
    parser.add_argument(
        "--hflip", help="Horizontal flip prob.", type=float, default=0.5
    )
    parser.add_argument(
        "--grad_clip", help="Gradient clipping value.", type=float, default=350
    )
    parser.add_argument(
        "--grad_skip", help="Skip update grad norm threshold.", type=float, default=500
    )
    parser.add_argument(
        "--accu_steps", help="Gradient accumulation steps.", type=int, default=1
    )
    parser.add_argument(
        "--beta", help="Max KL beta penalty weight.", type=float, default=1.0
    )
    parser.add_argument(
        "--beta_warmup_steps", help="KL beta penalty warmup steps.", type=int, default=0
    )
    parser.add_argument(
        "--kl_free_bits", help="KL min free bits constraint.", type=float, default=0.0
    )
    parser.add_argument(
        "--viz_freq", help="Steps per visualisation.", type=int, default=10000
    )
    parser.add_argument(
        "--eval_freq", help="Train epochs per validation.", type=int, default=5
    )
    # model
    parser.add_argument(
        "--vae",
        help="VAE model: simple/hierarchical.",
        type=str,
        default="hierarchical",
    )
    parser.add_argument(
        "--enc_arch",
        help="Encoder architecture config.",
        type=str,
        default="64b1d2,32b1d2,16b1d2,8b1d8,1b2",
    )
    parser.add_argument(
        "--dec_arch",
        help="Decoder architecture config.",
        type=str,
        default="1b2,8b2,16b2,32b2,64b2",
    )
    parser.add_argument(
        "--cond_prior",
        help="Use a conditional prior.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--widths",
        help="Number of channels.",
        nargs="+",
        type=int,
        default=[16, 32, 48, 64, 128],
    )
    parser.add_argument(
        "--bottleneck", help="Bottleneck width factor.", type=int, default=4
    )
    parser.add_argument(
        "--z_dim", help="Numver of latent channel dims.", type=int, default=16
    )
    parser.add_argument(
        "--z_max_res",
        help="Max resolution of stochastic z layers.",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--bias_max_res",
        help="Learned bias param max resolution.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--x_like",
        help="x likelihood: {fixed/shared/diag}_{gauss/dgauss}.",
        type=str,
        default="diag_dgauss",
    )
    parser.add_argument(
        "--std_init",
        help="Initial std for x scale. 0 is random.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--parents_x",
        help="Parents of x to condition on.",
        nargs="+",
        default=["mri_seq", "brain_volume", "ventricle_volume", "sex"],
    )
    parser.add_argument(
        "--concat_pa",
        help="Whether to concatenate parents_x.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--context_dim",
        help="Num context variables conditioned on.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--context_norm",
        help='Conditioning normalisation {"[-1,1]"/"[0,1]"/log_standard}.',
        type=str,
        default="log_standard",
    )
    parser.add_argument(
        "--q_correction",
        help="Use posterior correction.",
        action="store_true",
        default=False,
    )
    return parser
