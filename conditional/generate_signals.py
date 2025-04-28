### SOURCE: https://github.com/NVlabs/NVAE/blob/master/evaluate.py


import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import os

from torch.multiprocessing import Process
from torch.cuda.amp import autocast

from model_conditional_1d import AutoEncoder
import utils
from train_conditional_1d import init_processes


def set_bn(model, idx, bn_eval_mode, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()
        with autocast():
            for i in range(iter):
                if i % 10 == 0:
                    print('setting BN statistics iter %d out of %d' % (i+1, iter))
                model.sample(idx.cuda().float(), t)
        model.eval()


def generate(model, label, t):

    with torch.no_grad():
        torch.cuda.synchronize()
        with autocast():
            logits = model.sample(label.cuda().float(), t)

        output = model.decoder_output(logits)
        if args.num_input_channels == 8:
            output_img = output.sample(t)
        else:
            return NotImplementedError
        return output_img


def process_batches(model, t, num, label, count, batch_size, save_path):
    """Splits values into batches and calls generate() on each batch."""
    step = 0
    for count_i, start in enumerate(tqdm(range(0, count, batch_size))):
        count_i =int(min(batch_size, count - start))
        res = generate(model, torch.stack([torch.tensor(label)] * count_i), t).cpu().numpy()
        np.save(os.path.join(save_path,  f"label_{num}_sample_{step}.npy"), res)
        step += 1


def main(eval_args):

    # load a checkpoint
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu', weights_only=False)
    args = checkpoint['args']

    if not hasattr(args, 'ada_groups'):
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        args.num_mixture_dec = 10

    arch_instance = utils.get_arch_cells(args.arch_instance)

    labels = np.load(eval_args.labels_path)

    if eval_args.task_type == "imbalance":
        count = np.max(labels.sum(axis=0)) - labels.sum(axis=0)[1:]  # excluding 426783006 as the biggest class which doesn't need to be upsampled
        labels = []
        for i in range(1, 9):
            tmp = np.zeros(9)
            tmp[i] = 1
            labels.append(tmp)
    elif eval_args.task_type == "addition":
        labels, count = np.unique(labels, axis=0, return_counts=True)
        count *= 2
    else:
        raise NotImplementedError

    bn_eval_mode = not eval_args.readjust_bn

    for i, (label_i, count_i) in enumerate(tqdm(zip(labels, count))):

        t = float(random.choice(np.arange(0.6, 1.0, 0.1)))
        model = AutoEncoder(args, None, arch_instance, 9)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.cuda()
        set_bn(model, torch.stack([torch.tensor(label_i)] * eval_args.batch_size), bn_eval_mode, t=t, iter=eval_args.num_iters)

        if int(count_i) != 0:
            process_batches(model, t, i, label_i, count_i, eval_args.batch_size, eval_args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cNVAE-ECG signals generation')
    # experimental results
    parser.add_argument('--checkpoint', type=str, default='/tmp/expr/checkpoint.pt',
                        help='location of the checkpoint')
    parser.add_argument('--eval_mode', type=str, default='sample', choices=['sample', 'evaluate', 'evaluate_fid'],
                        help='evaluation mode. you can choose between sample or evaluate.')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--readjust_bn', action='store_true', default=False,
                        help='adding this flag will enable readjusting BN statistics.')
    parser.add_argument('--num_iw_samples', type=int, default=1000,
                        help='The number of IW samples used in test_ll mode.')
    parser.add_argument('--fid_dir', type=str, default='/tmp/fid-stats',
                        help='path to directory where fid related files are stored')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size used during evaluation. If set to zero, training batch size is used.')
    parser.add_argument('--num_iters', type=int, default=50,
                        help='Number of iterations for BN')
    # ECG
    parser.add_argument('--num_input_channels', type=int, default=1,
                help='Number of ECG leads')
    # DDP.
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6020',
                        help='port for master')
    # Generation info
    parser.add_argument('--labels_path', type=str, required=True,
                        help='path to PTBXL labels')
    parser.add_argument('--task_type', required=True, choices=['imbalance', 'addition'])
    parser.add_argument('--save_path', required=True, help='path to save generated samples by NVAE')

    args = parser.parse_args()
    args.save_path = os.path.join(args.save_path, "nvae", args.task_type)
    os.makedirs(args.save_path, exist_ok=False)

    size = args.world_size

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            p = Process(target=init_processes, args=(rank, size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args)