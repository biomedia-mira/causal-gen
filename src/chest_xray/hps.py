HPARAMS_REGISTRY = {}


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


chexpert64 = Hparams()
chexpert64.lr = 1e-3
chexpert64.bs = 32
chexpert64.wd = 0.1
chexpert64.z_dim = 16
chexpert64.input_res = 64
chexpert64.pad = 3
chexpert64.enc_arch = "64b3d2,32b31d2,16b15d2,8b7d2,4b3d4,1b2"
chexpert64.dec_arch = "1b2,4b4,8b8,16b16,32b32,64b4"
chexpert64.widths = [32, 64, 128, 256, 512, 1024]
HPARAMS_REGISTRY["chexpert64"] = chexpert64

chexpert192 = Hparams()
chexpert192.lr = 1e-3
chexpert192.bs = 16
chexpert192.wd = 0.1
chexpert192.z_dim = 16
chexpert192.input_res = 192
chexpert192.pad = 9
chexpert192.enc_arch = "192b1d2,96b3d2,48b7d2,24b11d2,12b7d2,6b3d6,1b2"
chexpert192.dec_arch = "1b2,6b4,12b8,24b12,48b8,96b4,192b2"
chexpert192.widths = [32, 64, 96, 128, 160, 192, 512]
HPARAMS_REGISTRY["chexpert192"] = chexpert192

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


def setup_hparams(parser):
    hparams = Hparams()
    args = parser.parse_known_args()[0]
    # args = parser.parse_args()
    valid_args = set(args.__dict__.keys())
    hparams_dict = HPARAMS_REGISTRY[args.hps].__dict__
    for k in hparams_dict.keys():
        if k not in valid_args:
            raise ValueError(f"{k} not in default args")
    parser.set_defaults(**hparams_dict)
    hparams.update(parser.parse_known_args()[0].__dict__)
    return hparams


def add_arguments(parser):
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument(
        "--data_dir", help="Data directory to load form.", type=str, default=""
    )
    parser.add_argument(
        "--csv_dir", help="CSV directory to load form.", type=str, default=""
    )
    parser.add_argument(
        "--use_dataset", help="Which dataset to use.", type=str, default=""
    )
    parser.add_argument("--hps", help="hyperparam set.", type=str, default="mimic192")
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
        "--input_res", help="Input image crop resolution.", type=int, default=192
    )
    parser.add_argument(
        "--input_channels", help="Input image num channels.", type=int, default=1
    )
    parser.add_argument("--pad", help="Input padding.", type=int, default=3)
    parser.add_argument(
        "--hflip", help="Horizontal flip prob.", type=float, default=0.5
    )
    parser.add_argument(
        "--grad_clip", help="Gradient clipping value.", type=float, default=100
    )
    parser.add_argument(
        "--grad_skip", help="Skip update grad norm threshold.", type=float, default=1000
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
    # parser.add_argument('--free_bits',
    #                     help='KL min free bits constraint.', type=float, default=1.)
    parser.add_argument(
        "--viz_freq", help="Steps per visualisation.", type=int, default=10000
    )
    parser.add_argument(
        "--eval_freq", help="Train epochs per validation.", type=int, default=5
    )
    # model
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
        default=["age", "race", "sex"],
    )
    parser.add_argument(
        "--context_dim",
        help="Num context variables conditioned on.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--context_norm",
        help='Conditioning normalisation {"[-1,1]"/"[0,1]"/log_standard}.',
        type=str,
        default="log_standard",
    )

    return parser
