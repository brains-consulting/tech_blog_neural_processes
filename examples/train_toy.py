import argparse
import torch
from torch import optim

import npmodel.utils as utils
from npmodel.model import NPModel
from npmodel.datasets.toydataset import GPCurvesReader, show_functions


def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                        help='number of epochs to train (default: 100000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='gpu number (default: 0), if no-cuda, ignore this option.')
    parser.add_argument('--seed', type=int, default=777, metavar='S',
                        help='random seed (default: 777)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--view', default=False, action="store_true",
                        help='show graphs on windows instead of saving to image files (default: False)')
    parser.add_argument('--visdom', default=False, action="store_true",
                        help='connecting the visdom server (default: False)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available() and (args.gpu >= 0)
    return args


def train(model, optimizer, epoch, npcfg):
    trainset = npcfg.trainset
    # trainset, _ = make_dataset(npcfg.train_gpr)
    testset, _ = make_dataset(npcfg.test_gpr)

    # set to train mode
    model.train()

    xC, yC, xT, yT = trainset
    optimizer.zero_grad()
    yhatT, sgm, loss = model(xC, yC, xT, yT)
    loss.backward()

    def loss_closure():
        return loss
    optimizer.step(loss_closure)
    loss_meter.update(loss.item())

    if epoch % 1000 == 0:
        B = min(xC.shape[0], 5)
        for bidx in range(B):
            model.plot_prediction(bidx, xC, yC, xT, yT)

    if epoch % 100 == 0:
        try:
            # if visdom server running, plot loss values
            plotter.plot("epoch", "loss", "train", "Epoch - Loss", [epoch], [loss_meter.avg], reset=False)
        except Exception as e:
            print(e)
        finally:
            loss_meter.reset()

    if epoch % npcfg.log_interval == 0:
        from datetime import datetime
        nw = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print(f"{nw} Train Epoch {epoch}/{npcfg.max_epoch} loss: {loss.item():.6f}")

        import pathlib
        print("convert trainset to images ...")
        file_name = f"img/train-{epoch:05d}.png"
        show_functions(file_name, *testset, yhatT, sgm, npcfg.view)
        p = pathlib.Path(file_name)
        p.parent.mkdir(parents=True, exist_ok=True)

        print("convert testset to images ...")
        file_name = f"img/test-{epoch:05d}.png"
        with torch.no_grad():
            yhatT, sgm = model.predict(*testset[:3])
        show_functions(file_name, *testset, yhatT, sgm, npcfg.view)


def make_dataset(gpr):
    data = gpr.generate_curves()
    ((context_x, context_y), target_x) = data.query
    target_y = data.target_y
    dataset = [context_x, context_y, target_x, target_y]
    for idx, d in enumerate(dataset):
        dataset[idx] = d.to(device)
    xC_size = context_x.shape[-1]
    yC_size = context_y.shape[-1]
    xT_size = target_x.shape[-1]
    yT_size = target_y.shape[-1]
    sizes = [xC_size, yC_size, xT_size, yT_size]
    return dataset, sizes


if __name__ == "__main__":
    args = get_args()
    utils.print_params(args, locals())

    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    train_gpr = GPCurvesReader(batch_size=args.batch_size, max_num_context=50, testing=False)
    test_gpr = GPCurvesReader(batch_size=1, max_num_context=50, testing=True)
    trainset, train_sizes = make_dataset(train_gpr)
    xC_size, yC_size, xT_size, yT_size = train_sizes

    import collections
    NPTrainConfig = collections.namedtuple(
        "NPTrainConfig", ("trainset", "train_gpr", "test_gpr", "log_interval", "max_epoch", "view")
    )
    npcfg = NPTrainConfig(
        trainset=trainset, train_gpr=train_gpr, test_gpr=test_gpr,
        log_interval=args.log_interval, max_epoch=args.epochs, view=args.view,
    )

    hidden_size = 128
    model_params = dict(
        xC_size=xC_size,
        yC_size=yC_size,
        xT_size=xT_size,
        yT_size=yT_size,
        z_size=hidden_size,
        embed_layers=[hidden_size]*3,
        latent_encoder_layers=[hidden_size]*1,
        deterministic_layers=[hidden_size]*4,
        decoder_layers=[hidden_size]*2 + [yT_size],
        use_deterministic_path=False,
    )
    model = NPModel(**model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    global plotter
    # plotter = utils.VisdomLinePlotter(env_name='np')
    if args.visdom:
        plotter = utils.VisdomLinePlotter(env_name='main')
        model.use_visdom()
    else:
        plotter = utils.FakeVisdomPlotter(env_name='main')


    global loss_meter
    loss_meter = utils.AverageMeter()

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, epoch, npcfg)

