### SOURCE: https://github.com/NVlabs/NVAE/blob/master/train.py


import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
import pickle
import torch.distributed as dist
from torch.multiprocessing import Process

from model_conditional_1d import AutoEncoder
import utils

import sys
sys.path.insert(0, '..')

from thirdparty.adamax import Adamax

from data.data_modules import ECGDataset
from torch.utils.data import DataLoader

from visualize import plot_ecg, compare_ecgs

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port  # '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()

def main(args):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    torch.set_default_dtype(torch.float64)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)
 
    with open(os.path.join(args.data_dir, "label2id.pickle"), "rb") as f:
        label2id = pickle.load(f)
    selected_classes = ['426783006', '39732003', '164873001', '164889003', '427084000', '270492004', '426177001', '164934002']

    train_dataset = ECGDataset(args.data_dir, "ptb-xl", label2id, selected_classes, option="train")
    valid_dataset = ECGDataset(args.data_dir, "ptb-xl", label2id, selected_classes, option="val")

    train_queue = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                              batch_size=args.batch_size, num_workers=4, drop_last=False)
    valid_queue = DataLoader(valid_dataset, shuffle=False, pin_memory=True,
                            batch_size=args.batch_size, num_workers=4, drop_last=False)

    args.num_total_iter = len(train_queue) * args.epochs
    print("len of train queue = {}, val queue = {}, total steps = {}".format(len(train_queue), len(valid_queue), args.num_total_iter))
    args.warmup_epochs = np.floor(args.epochs * args.percent_epochs / 100)
    warmup_iters = len(train_queue) * args.warmup_epochs

    arch_instance = utils.get_arch_cells(args.arch_instance)

    model = AutoEncoder(args, writer, arch_instance, num_classes = len(selected_classes))
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    logging.info('groups per scale: %s, total_groups: %d', model.groups_per_scale, sum(model.groups_per_scale))

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        cnn_optimizer = Adamax(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)

    # if load
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    print('model will be saved into {}'.format(checkpoint_file))
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 0

    for epoch in range(init_epoch, args.epochs):

        if epoch > args.warmup_epochs:
            cnn_scheduler.step()

        # Logging.
        logging.info('epoch %d', epoch)

        # Training.
        train_nelbo, global_step = train(train_queue, model, cnn_optimizer, global_step, warmup_iters, writer, logging, channels=args.num_input_channels) 
        logging.info('train_nelbo %f', train_nelbo)

        model.eval()
        # generate samples less frequently
        eval_freq = 5
        if epoch % eval_freq == 0 or epoch == (args.epochs - 1):
            with torch.no_grad():
                for t in [0.2, 0.5, 0.9]:
                    logits = model.sample(torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0.]]).cuda(), t)
                    output = model.decoder_output(logits)
                    output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample(t)
                    output_tiled = plot_ecg(output_img.cpu().squeeze(), args.num_input_channels)
                    writer.add_figure('test/generated_%0.1f' % t, output_tiled, global_step)

            valid_neg_log_p, valid_nelbo = test(valid_queue, model, num_samples=10, args=args, logging=logging, writer=writer, global_step=global_step, channels=args.num_input_channels)
            logging.info('valid_nelbo %f', valid_nelbo)
            logging.info('valid neg log p %f', valid_neg_log_p)
            writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch)
            writer.add_scalar('val/nelbo', valid_nelbo, epoch)

        save_freq = int(np.ceil(args.epochs / 100))
        print('save_freq = {}'.format(save_freq))
        
        if epoch % save_freq == 0 or epoch == (args.epochs - 1):
            if args.global_rank == 0:
                logging.info('saving the model.')
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                            'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                            'args': args, 'arch_instance': arch_instance, 'scheduler': cnn_scheduler.state_dict(),
                           }, checkpoint_file)
    # Final validation
    valid_neg_log_p, valid_nelbo = test(valid_queue, model, num_samples=100, args=args, logging=logging, writer=writer, global_step=global_step, channels=args.num_input_channels)
    logging.info('final valid nelbo %f', valid_nelbo)
    logging.info('final valid neg log p %f', valid_neg_log_p)
    writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch + 1)
    writer.add_scalar('val/nelbo', valid_nelbo, epoch + 1)
    writer.close()


def train(train_queue, model, cnn_optimizer, global_step, warmup_iters, writer, logging, channels):
    # A coefficient that gives greater KL-divergence weight to small (deeper) groups
    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    recon = utils.AvgrageMeter()
    kl = utils.AvgrageMeter()
    
    model.train()
    for x in tqdm(train_queue):
        image, label = x
        image = image.double().cuda()
        label = label.cuda()

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        cnn_optimizer.zero_grad()
        logits, log_q, log_p, kl_all, kl_diag = model(image, label.long())
        output = model.decoder_output(logits)
        # Coefficient dependent on the learning rate
        kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                  args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)

        recon_loss = utils.reconstruction_loss(output, image)


        balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=False, alpha_i=alpha_i)

        nelbo_batch = recon_loss + balanced_kl
        loss = torch.mean(nelbo_batch)
        bn_loss = model.batchnorm_loss()
        # get spectral regularization coefficient (lambda)
        if args.weight_decay_norm_anneal:
            assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
            wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
            wdn_coeff = np.exp(wdn_coeff)
        else:
            wdn_coeff = args.weight_decay_norm

        loss += bn_loss * wdn_coeff

        loss.backward()
        cnn_optimizer.step()
        
        nelbo.update(loss.data, 1)
        recon.update(torch.mean(recon_loss).data, 1)
        kl.update(torch.mean(balanced_kl).data, 1)

        if (global_step + 0) % 84 == 0:
            if (global_step + 0) % 840 == 0:  # reduced frequency
                n = 3
                x_img = image[:n]
                output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()
                output_img = output_img[:n]
                for i in range(n):
                    writer.add_figure('train/reconstruction for sample %d' % i, compare_ecgs(x_img[i,:,:].cpu(), output_img[i,:,:].cpu()), global_step)


            # norm
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/recon_avg', recon.avg, global_step)
            writer.add_scalar('train/kl_avg', kl.avg, global_step)

            writer.add_scalar('train/lr', cnn_optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), global_step)

            writer.add_scalar('train/recon_iter', torch.mean(utils.reconstruction_loss(output, image)), global_step)


            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                writer.add_scalar('kl/kl_diag_%d' % i, kl_diag_i.sum(), global_step)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], global_step)
            writer.add_scalar('kl/total_active', total_active, global_step)

        global_step += 1
    return nelbo.avg, global_step


def test(valid_queue, model, num_samples, args, logging, writer, global_step, channels):
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()
    for step, x in enumerate(tqdm(valid_queue)):

        image, label = x
        image = image.double().cuda()

        with torch.no_grad():
            nelbo, log_iw = [], []  # importance weighted
            logits, log_q, log_p, kl_all, kl_diag = model(image, label.long())
            output = model.decoder_output(logits)
            recon_loss = utils.reconstruction_loss(output, image)


            balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
            nelbo_batch = recon_loss + balanced_kl

            nelbo.append(nelbo_batch)
            log_iw.append(utils.log_iw(output, image, log_q, log_p))

            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

        nelbo_avg.update(nelbo.data, image.size(0))
        neg_log_p_avg.update(- log_p.data, image.size(0))

    n = 3
    x_img = image[:n]
    output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()
    output_img = output_img[:n]
    for i in range(n):
        writer.add_figure('test/reconstruction for number %d' % i, compare_ecgs(x_img[i,:,:].cpu(), output_img[i,:,:].cpu()), global_step)


    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg


def create_generator_vae(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.sample(batch_size, 1.0)
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
        yield output_img.float()


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port  # '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--data_dir', type=str, default='/home/jovyan/isviridov/gans/gan_ecg/data/',
                        help='location of the data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=5e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=1e-2,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--percent_epochs', type=int, default=1,
                        help='fraction of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=1,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6020',
                        help='port for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    # Debug
    parser.add_argument('--input_size', type=int, default=5000,
                    help='Signal input size'),
    # ECG
    parser.add_argument('--num_input_channels', type=int, default=1,
                    help='Number of ECG leads')
    parser.add_argument('--name', type=str, required=True,
                    help='experiment name')
    parser.add_argument('--focal', action='store_true', default=False,
                    help='use focal loss in ce?')
    args = parser.parse_args()
    args.save = args.root + args.name
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args)


