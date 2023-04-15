import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", default="/data", help="position of folder which has the downloaded dataset")
parser.add_argument("--data_dir", default="sub_KSL", help="folder name of downloaded dataset")

parser.add_argument("--batch_size", type=float, default=512, help="folder name of downloaded dataset")
parser.add_argument("--shuffle", type=str2bool, default=True, help="folder name of downloaded dataset")
parser.add_argument("--input_w", type=int, default=112, help="folder name of downloaded dataset")
parser.add_argument("--input_h", type=int, default=112, help="folder name of downloaded dataset")

args = parser.parse_args()
