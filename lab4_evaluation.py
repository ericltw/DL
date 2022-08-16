import argparse
from lab4_dataset import bair_robot_pushing_dataset
from lab4_utils import finn_eval_seq, pred
import lab4_utils
import numpy as np
import os
import random
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--data_root', default='lab4_data/processed_data', help='root directory for data')
    parser.add_argument('--model_path', default='', help='path to model')
    parser.add_argument('--log_dir', default='lab4_evaluation', help='directory to save generations to')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--num_threads', type=int, default=1, help='number of data loading threads')
    parser.add_argument('--n_sample', type=int, default=3, help='number of samples')

    return parser.parse_args()


def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w + 2 * pad + 30, w + 2 * pad))
    if color == 'red':
        px[0] = 0.7
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w + pad, pad:w + pad] = x
    else:
        px[:, pad:w + pad, pad:w + pad] = x
    return px


def make_gifs(x, cond, idx, name):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()

    output_seq = [x[0]]
    x_in = x[0]

    for i in range(1, args.n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()

        # TODO
        if args.last_frame_skip or i < args.n_past:
            h, skip = h
        else:
            h, _ = h

        h = h.detach()
        _, z_t, _ = posterior(h_target)

        if i < args.n_past:
            frame_predictor(torch.cat([cond[i - 1], h, z_t], 1))
            output_seq.append(x[i])
            x_in = x[i]
        else:
            g_t = frame_predictor(torch.cat([cond[i - 1], h, z_t], 1)).detach()
            x_in = decoder([g_t, skip]).detach()
            output_seq.append(x_in)

    # Init variables.
    n_sample = args.n_sample
    ssim = np.zeros((args.batch_size, n_sample, args.n_future))
    psnr = np.zeros((args.batch_size, n_sample, args.n_future))

    progress = tqdm(total=n_sample)
    all_gen = []

    for number_of_sample in range(n_sample):
        progress.update(1)

        gen_seq = []
        gt_seq = []

        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()

        x_in = x[0]
        all_gen.append([])
        all_gen[number_of_sample].append(x_in)
        for i in range(1, args.n_eval):
            h = encoder(x_in)
            if args.last_frame_skip or i < args.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < args.n_past:
                h_target = encoder(x[i])[0].detach()
                _, z_t, _ = posterior(h_target)
            else:
                z_t = torch.randn(args.batch_size, args.z_dim).cuda()
            if i < args.n_past:
                frame_predictor(torch.cat([cond[i - 1], h, z_t], 1))
                x_in = x[i]
                all_gen[number_of_sample].append(x_in)
            else:
                h = frame_predictor(torch.cat([cond[i - 1], h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in)
                gt_seq.append(x[i])
                all_gen[number_of_sample].append(x_in)
        _, ssim[:, number_of_sample, :], psnr[:, number_of_sample, :] = finn_eval_seq(gt_seq, gen_seq)

    # ssim
    for i in range(args.batch_size):
        gifs = [[] for t in range(args.n_eval)]
        text = [[] for t in range(args.n_eval)]
        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(n_sample) for s in range(3)]

        for t in range(args.n_eval):
            # gt
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            # posterior
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(output_seq[t][i], color))
            text[t].append('Approx.\nposterior')
            # best
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best SSIM')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s + 1))

        fname = '%s/%s_%d.gif' % (args.log_dir, name, idx + i)
        lab4_utils.save_gif_with_text(fname, gifs, text)


if __name__ == '__main__':
    # Parse arguments.
    args = parse_args()

    # Create directory for recording logs.
    os.makedirs('%s' % args.log_dir, exist_ok=True)

    args.n_eval = args.n_past + args.n_future
    args.max_step = args.n_past + args.n_future

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.cuda.FloatTensor

    # ---------------- load the models  ----------------
    modules = torch.load(args.model_path)
    frame_predictor = modules['frame_predictor']
    posterior = modules['posterior']
    encoder = modules['encoder']
    decoder = modules['decoder']

    # Set module in evaluation mode.
    frame_predictor.eval()
    posterior.eval()
    encoder.eval()
    decoder.eval()

    # Set model batch size.
    frame_predictor.batch_size = args.batch_size
    posterior.batch_size = args.batch_size

    # ---------------- transfer to gpu ----------------
    frame_predictor.cuda()
    posterior.cuda()
    encoder.cuda()
    decoder.cuda()

    # ---------------- set the args ----------------
    args.last_frame_skip = modules['args'].last_frame_skip
    args.g_dim = modules['args'].g_dim
    args.z_dim = modules['args'].z_dim


    print(args)

    # ---------------- load a dataset ----------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                             num_workers=args.num_threads,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    test_iterator = iter(test_loader)

    # plot test
    device = 'cuda'
    psnr_list = []
    for i, (test_seq, test_cond) in enumerate(tqdm(test_loader)):
        test_seq = test_seq.permute(1, 0, 2, 3, 4).to(device)
        test_cond = test_cond.permute(1, 0, 2).to(device)
        pred_seq = pred(test_seq, test_cond, modules, args, device)
        _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
        psnr_list.append(psnr)
    ave_psnr = np.mean(np.concatenate(psnr_list))
    print(f'Test psnr: {ave_psnr:.5f}')

    test_iterator = iter(test_loader)
    test_seq, test_cond = next(test_iterator)
    test_seq = test_seq.permute(1, 0, 2, 3, 4).to(device)
    test_cond = test_cond.permute(1, 0, 2).to(device)
    make_gifs(test_seq, test_cond, 0, 'test')
