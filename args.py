import argparse

audio_len=640
def get_args():
    """
    Function for handling command line arguments

    :return: parsed   command line arguments
    """
    parser = argparse.ArgumentParser(description='WaveNet')

    # experiment
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed')
    parser.add_argument('--experiment', type=str, default="my_model", metavar='EXP',
                        help='experiment name')

    # model
    parser.add_argument('--num-blocks', type=int, default=4, metavar='NUM_BLOCKS',
                        help='number of blocks')
    parser.add_argument('--num-layers', type=int, default=6, metavar='NUM_LAYERS',
                        help='number of layers')

    parser.add_argument('--ch-residual', type=int, default=32, metavar='CH_RESIDUAL',
                            help='residual channels')
    parser.add_argument('--ch-dilation', type=int, default=32, metavar='CH_DILATION',
                            help='dilation channels')
    parser.add_argument('--ch-skip', type=int, default=64, metavar='CH_SKIP',
                            help='skip channels')
    parser.add_argument('--ch-end', type=int, default=10, metavar='CH_END',
                            help='end channels')

    parser.add_argument('--kernel-size', type=int, default=3, metavar='KERNEL_SIZE',
                            help='number of blocks')

    # dataset
    parser.add_argument('--win-len', type=int, default=128, metavar='WIN_LEN',
                        help='window length')
    parser.add_argument('--out-len', type=int, default=audio_len, metavar='OUT_LEN',
                        help='output length')
    parser.add_argument('--emb-dim', type=int, default=64, metavar='EMB_DIM',
                        help='embedding dimension')

    parser.add_argument('--inp-len', type=int, default=audio_len, metavar='INP_LEN',
                        help='train input length')
    # training
    parser.add_argument('--batch-size', type=int, default=64, metavar='BATCH_SIZE',
                        help='batch size')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate')

    # Argument parsing
    return parser.parse_args()