import argparse
from lab4_dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from lab4_utils import init_weights, kl_criterion, plot_pred, plot_GIF, plot_pred, finn_eval_seq, pred
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./lab4_logs', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='./lab4_data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=0,
                        help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0,
                        help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0,
                        help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3,
                        help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128,
                        help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=1, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true',
                        help='if true, skip connections go between frame t and frame t+t rather than last ground truth'
                             ' frame')
    parser.add_argument('--cuda', default=True, action='store_true')

    args = parser.parse_args()
    return args


def train(x, cond, modules, optimizer, kl_anneal, args):
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()
    mse_criterion = nn.MSELoss()

    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    mse = 0
    kld = 0
    use_teacher_forcing = True if random.random() < args.tfr else False

    # Rearrange the original tensors.
    x = x.permute(1, 0, 2, 3, 4)
    cond = cond.permute(1, 0, 2)

    # Compute h sequence.
    h_seq = [modules['encoder'](x[i]) for i in range(args.n_past + args.n_future)]

    for i in range(1, args.n_past + args.n_future):
        # Target image.
        h_target = h_seq[i][0]

        # TODO
        if args.last_frame_skip or i < args.n_past:
            h, skip = h_seq[i - 1]
        else:
            h = h_seq[i - 1][0]

        # Forward encoder.
        z_t, mu, log_var = modules['posterior'](h_target)

        # Forward decoder.
        h_pred = modules['frame_predictor'](torch.cat([cond[i - 1], h, z_t], 1))
        x_pred = modules['decoder']([h_pred, skip])

        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, log_var, args)

        # For decoder/encoder input x(t-1)
        if not use_teacher_forcing:
            h_seq[i] = modules['encoder'](x_pred)

    beta = kl_anneal.get_beta()
    loss = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (
            args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)


class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.values = np.ones(args.niter) * 1.0
        self.iteration = 0

        # Cyclical mode.
        if args.kl_anneal_cyclical:
            cycle_length = args.niter / args.kl_anneal_cycle
            delta = (1.0 - 0) / (cycle_length * 0.5)

            for num_of_cycle in range(args.kl_anneal_cycle):
                value, iteration = 0.0, 0
                while value <= 1.0 and int(iteration + num_of_cycle * cycle_length) < args.niter:
                    self.values[int(iteration + num_of_cycle * cycle_length)] = value
                    value += delta
                    iteration += 1
        # Monotonic mode.
        else:
            delta = (1.0 - 0) / (args.niter * 0.25)
            value, iteration = 0.0, 0
            while value <= 1.0 and int(iteration) < args.niter:
                self.values[iteration] = value
                value += delta
                iteration += 1

    def update(self):
        self.iteration += 1

    def get_beta(self):
        return self.values[self.iteration]


def main():
    # Parse arguments.
    args = parse_args()

    # Get device.
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'

    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch
    assert 0 <= args.tfr_decay_step <= 1

    # Load arguments and continue training from checkpoint.
    if args.model_dir != '':
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    # Init parameters.
    else:
        name = 'rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-' \
               'last_frame_skip=%s-beta=%.7f' \
               % (
                   args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future,
                   args.lr,
                   args.g_dim, args.z_dim, args.last_frame_skip, args.beta)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 0

    # Create directory for recording logs.
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Remove log file if the log file exists.
    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))

    print(args)

    # Write input arguments to log file.
    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # --------- build the models  ------------------------------------
    # Load models and continue training from checkpoint.
    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
    # Init models.
    else:
        # TODO
        # frame_predictor is for decoder side.
        frame_predictor = lstm(args.g_dim + args.z_dim + 7, args.g_dim, args.rnn_size, args.predictor_rnn_layers,
                               args.batch_size, device)
        # posterior is for encoder side.
        posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.batch_size,
                                  device)
        # Apply initial weights to models.
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)

    # Load models and continue training from checkpoint.
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    # Init models.
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)

    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    train_loader = DataLoader(train_data,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    # train_iterator = iter(train_loader)

    validate_data = bair_robot_pushing_dataset(args, 'validate')
    validate_loader = DataLoader(validate_data,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 pin_memory=True)
    validate_iterator = iter(validate_loader)

    # --------- optimizers ------------------------------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(posterior.parameters()) + list(encoder.parameters()) + list(
        decoder.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))

    # --------- kl_annealing ------------------------------------
    kl_anneal = kl_annealing(args)

    # --------- model dictionary ------------------------------------
    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder,
    }

    # --------- training loop ------------------------------------
    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    args.epoch_size = len(train_loader)

    for epoch in range(start_epoch, start_epoch + niter):
        # Set module in training mode.
        frame_predictor.train()
        posterior.train()
        encoder.train()
        decoder.train()

        # Init parameters.
        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        # Training process.
        for i, (seq, cond) in enumerate(tqdm(train_loader)):
            seq, cond = seq.to(device), cond.to(device)

            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld

        # Update iteration value in kl_anneal object.
        kl_anneal.update()

        # Update teacher forcing ratio.
        if epoch >= args.tfr_start_decay_epoch:
            total_delta = 1.0 - args.tfr_lower_bound
            total_decay_epoch = args.niter - args.tfr_start_decay_epoch
            slope = total_delta / total_decay_epoch

            tfr = 1.0 - (epoch - args.tfr_start_decay_epoch) * slope
            args.tfr = min(1, max(args.tfr_lower_bound, tfr))

        progress.update(1)

        # Write epoch summary to train_record file.
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (
                epoch, epoch_loss / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))

        # Set module in evaluation mode.
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()

        # Validate model every 5 epochs.
        if epoch % 5 == 0:
            psnr_list = []

            # Iterate for computing psnr for every validated sequence.
            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)

                validate_seq = validate_seq.permute(1, 0, 2, 3, 4).to(device)
                validate_cond = validate_cond.permute(1, 0, 2).to(device)
                pred_seq = pred(validate_seq, validate_cond, modules, args, device)
                _, _, psnr = finn_eval_seq(validate_seq[args.n_past:], pred_seq[args.n_past:])
                psnr_list.append(psnr)
            ave_psnr = np.mean(np.concatenate(psnr))

            # Write average psnr result to train_record.
            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(
                    ('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

            # If the average psnr is larger than the best psnr, save the model.
            if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)

        # Plot pred every 20 epochs.
        if epoch % 20 == 0:
            try:
                validate_seq, validate_cond = next(validate_iterator)
            except StopIteration:
                validate_iterator = iter(validate_loader)
                validate_seq, validate_cond = next(validate_iterator)

            validate_seq = validate_seq.permute(1, 0, 2, 3, 4).to(device)
            validate_cond = validate_cond.permute(1, 0, 2).to(device)
            plot_GIF(validate_seq, validate_cond, modules, epoch, args)
            plot_pred(validate_seq, validate_cond, modules, epoch, args)


if __name__ == '__main__':
    main()
