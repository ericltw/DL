import argparse
from lab5_dataset import iclevrDataset
from lab5_evaluator import evaluation_model
from models import ACGAN
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--ngf', type=int, default=300, help="feature channels of generator")
    parser.add_argument('--ndf', type=int, default=64, help="feature channels of discriminator")
    parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--nc", type=int, default=100, help="number of condition embedding dim")
    parser.add_argument('--outf', default='inference_image', help='folder to output images and model checkpoints')
    parser.add_argument('--resume', default='', help='path to resume model weight')
    parser.add_argument('--test_file', default='test.json', help='test json')
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.outf, exist_ok=True)

    # load generator and discriminator weights
    generator = ACGAN.Generator(args).to(device)
    discriminator = ACGAN.Discriminator(args).to(device)
    generator.load_state_dict(torch.load(os.path.join(args.resume, 'netG.pth')))
    discriminator.load_state_dict(torch.load(os.path.join(args.resume, 'netD.pth')))

    test_dataloader = DataLoader(
        iclevrDataset(mode='test', root='iclevr', test_file=args.test_file),
        batch_size=args.batch_size,
        shuffle=False
    )
    # Initialize evaluator
    evaluator = evaluation_model()
    discriminator.eval()
    generator.eval()
    with torch.no_grad():
        avg_acc = 0
        for sample in range(10):
            for i, cond in enumerate(test_dataloader):
                cond = cond.to(device)
                batch_size = cond.size(0)
                noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                fake_image = generator(noise, cond)
                save_image(fake_image.detach(),
                           '%s/fake_test_sample%d.png' % (args.outf, sample),
                           normalize=True)
                acc = evaluator.eval(fake_image, cond)
            print(f'Sample {sample + 1}: {acc * 100:.2f}%')
            avg_acc += acc
        avg_acc /= 10
        print(f'Average acc: {avg_acc * 100:.2f}%')


if __name__ == '__main__':
    main()
