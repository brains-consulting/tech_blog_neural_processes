import argparse
import pathlib
import collections
import traceback

import torch
from torch import optim

import npmodel.utils as utils
from npmodel.model import NPModel
from npmodel.datasets.mnists import NPMnistReader, NPBatches, show_yimages


def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='F',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='gpu number (default: 0), if no-cuda, ignore this option.')
    parser.add_argument('--seed', type=int, default=777, metavar='S',
                        help='random seed (default: 777)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "fashion"], metavar='S',
                        help='dataset name like "mnist", "fashion-mnist", default: "mnist"')
    parser.add_argument('--fix-iter', type=int, default=1000, metavar='N',
                        help='the number of training iteration with a fixed sampling dataset/imageset batch'
                             ', if negative use whole dataset (default: 1000)')
    parser.add_argument('--view', default=False, action="store_true",
                        help='show graphs on windows instead of saving to image files (default: False)')
    parser.add_argument('--visdom', default=False, action="store_true",
                        help='connecting the visdom server (default: False)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available() and (args.gpu >= 0)
    return args


class Trainer(object):
    def __init__(self, train_params):
        self.train_params = train_params

        batch_size, device = train_params.batch_size, train_params.device
        seed = train_params.seed
        params = dict(
            shuffle=True, seed=seed, mnist_type="mnist", fix_iter=train_params.fix_iter, device=device,
        )
        self.train_reader = NPMnistReader(batch_size=batch_size, testing=False, **params)
        self.test_reader = NPMnistReader(batch_size=batch_size, testing=True, **params)

        self.model = None
        self.optimizer = None

        if train_params.visdom:
            self.plotter = utils.VisdomLinePlotter(env_name=self.train_params.env_name)
        else:
            self.plotter = utils.FakeVisdomPlotter(env_name=self.train_params.env_name)
        self.loss_meter = utils.AverageMeter()

    def get_dims(self):
        itm = next(iter(self.train_reader))
        return itm.dims()   # xC_size, yC_size, xT_size, yT_size

    def run_train(self, model, optimizer):
        for epoch in range(1, self.train_params.max_epoch + 1):
            self.train_epoch(epoch, model, optimizer)

    def train_epoch(self, epoch, model, optimizer):
        # hold as local variable
        train_params = self.train_params

        # keep on self
        self.model = model
        self.optimizer = optimizer

        # set to train mode
        model.train()

        test_itr = iter(self.test_reader)
        for b, train_itm in enumerate(self.train_reader):
            train_batches = train_itm.batches()   # xC, yC, xT, yT
            optimizer.zero_grad()
            yhatT, sgm, loss = model(*train_batches)
            loss.backward()

            def loss_closure():
                return loss
            optimizer.step(loss_closure)
            self.loss_meter.update(loss.item())

        # by epoch
        try:
            from datetime import datetime
            nw = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            print(f"{nw} Train Epoch {epoch:05d}/{train_params.max_epoch:05d} loss: {self.loss_meter.avg:.6f}")

            # if visdom server running, plot loss values
            self.plotter.plot("epoch", "loss", "train", "Epoch - Loss", [epoch], [self.loss_meter.avg], reset=False)
        except Exception as e:
            print(traceback.format_exc(chain=e))
        finally:
            self.loss_meter.reset()     # by every epoch

        if epoch % train_params.log_interval == 0:
            print(f"convert trainset to images ...")
            train_itr = iter(self.train_reader)
            train_itm = next(train_itr)
            self.save_images(epoch, "train", train_itm)

            print(f"convert testset to images ...")
            try:
                test_itm = next(test_itr)
            except StopIteration:
                test_itr = iter(self.test_reader)
                test_itm = next(test_itr)
            self.save_images(epoch, "test", test_itm)

    def save_images(self, epoch: int, name: str, batch_item: NPBatches):
        assert name in ["train", "test"]
        batches = batch_item.batches()
        file_name = f"img/{name}-{epoch:05d}_{batch_item.idx:05d}.png"
        p = pathlib.Path(file_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            yhatT, sgm = self.model.predict(*batches[:3])
            xC, yC, xT, yT = batches
            img_yC = self.train_reader.convert_to_img(xC, yC)
            img_yT = self.train_reader.convert_to_img(xT, yT)
            img_yhat = self.train_reader.convert_to_img(xT, yhatT)
            show_yimages(img_yC, img_yT, img_yhat, "yC/yhatT/yT", file_name, self.train_params.view)


if __name__ == "__main__":
    args = get_args()
    utils.print_params(args, locals())
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    TrainParameters = collections.namedtuple(
        "TrainParameters",
        ("batch_size", "env_name", "log_interval", "max_epoch", "fix_iter", "visdom", "seed", "view", "device")
    )
    train_params = TrainParameters(
        batch_size=args.batch_size, env_name="main",
        log_interval=args.log_interval, max_epoch=args.epochs, fix_iter=args.fix_iter, visdom=args.visdom,
        seed=args.seed, view=args.view, device=device
    )
    trainer = Trainer(train_params)
    xC_dim, yC_dim, xT_dim, yT_dim = trainer.get_dims()

    hidden_size = 128
    model_params = dict(
        xC_size=xC_dim,
        yC_size=yC_dim,
        xT_size=xT_dim,
        yT_size=yT_dim,
        z_size=hidden_size,
        embed_layers=[512, 256, 128],
        latent_encoder_layers=[128, 256],
        deterministic_layers=[hidden_size]*4,
        use_deterministic_path=False,
        decoder_layers=[128, 256, 512] + [yT_dim],
        decout_fn=torch.sigmoid,
    )

    model = NPModel(**model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer.run_train(model, optimizer)
