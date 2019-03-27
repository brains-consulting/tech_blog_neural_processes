import numpy
from visdom import Visdom
from abc import ABCMeta, abstractmethod


def print_params(args, g=globals()):
    name = "params"
    for nm, vr in g.items():
        if id(vr) == id(args):
            name = nm
            break

    print("-" * 25)
    for k, v in sorted(args.__dict__.items()):
        if k.startswith("_"):
            continue
        if k.startswith("no"):
            continue
        print(f"{name}.{k}: {v}")
    print("-" * 5)


class VisdomPlotterInterface(metaclass=ABCMeta):
    @abstractmethod
    def plot(self, x_label, y_label, trace_name, title_name, x, y, reset=False):
        raise NotImplementedError()

    @abstractmethod
    def scatter(self, x, y, y_label, trace_name, color=(0, 0, 0)):
        raise NotImplementedError()


class VisdomLinePlotter(VisdomPlotterInterface):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8097, server="localhost"):
        self.viz = Visdom(port=port, server=server)
        self.env = env_name
        self.plotted_windows = {}
        self.traces = []

    def plot(self, x_label, y_label, trace_name, title_name, x, y, reset=False):
        # X=x if hasattr(x, "__iter__") else numpy.array([x]),
        # Y=y if hasattr(y, "__iter__") else numpy.array([y]),
        if trace_name not in self.traces:
            self.traces.append(trace_name)
        params = dict(
            X=x,
            Y=y,
            env=self.env,
            name=trace_name,
        )
        extra = dict(opts=dict(legend=[trace_name],
                               title=title_name,
                               xlabel=x_label,
                               ylabel=y_label,
                               ))
        if y_label in self.plotted_windows:  # just at the first time
            _extra = dict(
                win=self.plotted_windows[y_label],
            )
            if not reset:
                _extra = dict(
                    win=self.plotted_windows[y_label],
                    update='append',
                )
            extra.update(_extra)
        params.update(extra)
        self.plotted_windows[y_label] = self.viz.line(**params)

    def scatter(self, x, y, y_label, trace_name, color=(0, 0, 0)):
        color = numpy.array(color).reshape(-1, 3)
        win = self.plotted_windows[y_label]
        self.viz.scatter(X=x, Y=y,
                         opts=dict(markersize=10, markercolor=color,),
                         name=trace_name,
                         update='append',
                         win=win)


class FakeVisdomPlotter(VisdomPlotterInterface):
    """Fake Plotter"""
    def __init__(self, env_name='main', port=8097, server="localhost"):
        self.env = env_name

    def plot(self, *args, **kwargs):
        pass    # Do Nothing

    def scatter(self, args, **kwargs):
        pass    # Do Nothing


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = -1
        self.avg = -1
        self.sum = -1
        self.count = -1
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

