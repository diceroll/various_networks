import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path

import chainer
import chainer.functions as F
import chainer.links as L
import cupy
import numpy as np
from chainer import iterators, optimizers, serializers
from chainer.training import StandardUpdater, Trainer, extensions

from food_101_dataset import Food101Dataset
from models.aa_resnet import AAResNet50
from models.gc_resnet import GCResNet50
from models.se_res2net import SERes2Net50
from models.se_resnet import SEResNet50


def main():
    parser = argparse.ArgumentParser(description='training mnist')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--report_trigger', '-rt', type=str, default='1e',
                        help='Interval for reporting (Ex.100i/1e)')
    parser.add_argument('--save_trigger', '-st', type=str, default='1e',
                        help='Interval for saving the model (Ex.100i/1e)')
    parser.add_argument('--load_model', '-lm', type=str, default=None,
                        help='Path of the model object to load')
    parser.add_argument('--load_optimizer', '-lo', type=str, default=None,
                        help='Path of the optimizer object to load')
    args = parser.parse_args()

    if not Path('output').exists():
        Path('output').mkdir()
    start_time = datetime.now()
    save_dir = Path('output/{}'.format(start_time.strftime('%Y%m%d_%H%M')))

    random.seed(args.seed)
    np.random.seed(args.seed)
    cupy.random.seed(args.seed)
    chainer.config.cudnn_deterministic = True

    model = L.Classifier(SEResNet50(n_class=101))
    # model = L.Classifier(SERes2Net50(n_class=101))
    # model = L.Classifier(GCResNet50(n_class=101))
    # model = L.Classifier(AAResNet50(n_class=101))

    if args.load_model is not None:
        serializers.load_npz(args.load_model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam(
        alpha=1e-3, weight_decay_rate=1e-4, amsgrad=True)
    optimizer.setup(model)
    if args.load_optimizer is not None:
        serializers.load_npz(args.load_optimizer, optimizer)

    augmentation = {
        'HorizontalFlip': {'p': 0.5},
        'PadIfNeeded': {'p': 1.0, 'min_height': 512, 'min_width': 512},
        'Rotate': {'p': 1.0, 'limit': 15, 'interpolation': 1},
        'Resize': {'p': 1.0, 'height': 248, 'width': 248, 'interpolation': 2},
        'RandomScale': {'p': 1.0, 'scale_limit': 0.09, 'interpolation': 2},
        'RandomCrop': {'p': 1.0, 'height': 224, 'width': 224},
    }
    resize = {
        'PadIfNeeded': {'p': 1.0, 'min_height': 512, 'min_width': 512},
        'Resize': {'p': 1.0, 'height': 224, 'width': 224, 'interpolation': 2}
    }

    sl = slice(0, None, 5)
    train_data = Food101Dataset(augmentation=augmentation, drop_index=sl)
    valid_data = Food101Dataset(augmentation=resize, index=sl)

    train_iter = iterators.SerialIterator(train_data, args.batchsize)
    valid_iter = iterators.SerialIterator(
        valid_data, args.batchsize, repeat=False, shuffle=False)

    updater = StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = Trainer(updater, (args.epoch, 'epoch'), out=save_dir)

    report_trigger = (
        int(args.report_trigger[:-1]),
        'iteration' if args.report_trigger[-1] == 'i' else 'epoch')
    trainer.extend(extensions.LogReport(trigger=report_trigger))
    trainer.extend(extensions.Evaluator(
        valid_iter, model, device=args.gpu),
        name='val', trigger=report_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/accuracy', 'val/main/loss',
         'val/main/accuracy', 'elapsed_time']), trigger=report_trigger)
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key=report_trigger[1],
        marker='.', file_name='loss.png', trigger=report_trigger))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key=report_trigger[1],
        marker='.', file_name='accuracy.png', trigger=report_trigger))

    save_trigger = (
        int(args.save_trigger[:-1]),
        'iteration' if args.save_trigger[-1] == 'i' else 'epoch')
    trainer.extend(extensions.snapshot_object(
        model, filename='model_{0}-{{.updater.{0}}}.npz'
        .format(save_trigger[1])), trigger=save_trigger)
    trainer.extend(extensions.snapshot_object(
        optimizer, filename='optimizer_{0}-{{.updater.{0}}}.npz'
        .format(save_trigger[1])), trigger=save_trigger)
    trainer.extend(extensions.ProgressBar())

    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir()

    # Write parameters text
    with open(save_dir / 'train_params.txt', 'w') as f:
        f.write('model: {}\n'.format(model.predictor.__class__.__name__))
        f.write('n_epoch: {}\n'.format(args.epoch))
        f.write('batch_size: {}\n'.format(args.batchsize))
        f.write('seed: {}\n'.format(args.seed))
        f.write('n_data_train: {}\n'.format(len(train_data)))
        f.write('n_data_val: {}\n'.format(len(valid_data)))
        f.write('augmentation: \n')
        for k, v in augmentation.items():
            f.write('  {}: {}\n'.format(k, v))

    trainer.run()


if __name__ == '__main__':
    main()
