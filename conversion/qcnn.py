#!/usr/bin/env python3
"""4-bit QNN VGG-16 (half width) for CIFAR-10: LSQ weights and activations, training and evaluation."""

import argparse, math, os, time
import torch, torch.nn as nn, torch.nn.functional as F

CFG = [32, 32, "M", 64, 64, "M", 128, 128, 128, "M", 256, 256, 256, "M", 256, 256, 256, "M"]
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(os.path.dirname(HERE), "models", "vgg16h", "checkpoint.pt")
DATA = os.path.expanduser("~/datasets")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grad_scale(x, s):
    return (x - x * s).detach() + x * s


def round_ste(x):
    return (x.round() - x).detach() + x


class LSQ(nn.Module):
    """Learned step size quantizer.  signed: [-2^(b-1), 2^(b-1)-1], else [0, 2^b-1]."""

    def __init__(self, bits, signed, channels=1):
        super().__init__()
        self.qn = -(2 ** (bits - 1)) if signed else 0
        self.qp = 2 ** (bits - 1) - 1 if signed else 2**bits - 1
        self.step = nn.Parameter(torch.ones(channels))
        self.register_buffer("init", torch.zeros(1))
        self.enabled = True

    def levels(self, x):
        """Integer levels (no STE) -- used by the analysis."""
        s = self.step.view(-1, *([1] * (x.dim() - 1))) if self.step.numel() > 1 else self.step
        return torch.clamp(torch.round(x / s), self.qn, self.qp)

    def forward(self, x):
        if not self.enabled:
            return x
        if self.init.item() == 0:
            with torch.no_grad():
                if self.step.numel() > 1:
                    m = x.detach().abs().flatten(1).mean(1)
                else:
                    m = x.detach().abs().mean().view(1)
                self.step.copy_(2 * m / math.sqrt(self.qp) + 1e-8)
                self.init.fill_(1)
        g = 1.0 / math.sqrt(x.numel() / self.step.numel() * self.qp)
        s = grad_scale(self.step, g)
        s = s.view(-1, *([1] * (x.dim() - 1))) if self.step.numel() > 1 else s
        return round_ste(torch.clamp(x / s, self.qn, self.qp)) * s


class QConv(nn.Conv2d):
    def __init__(self, cin, cout):
        super().__init__(cin, cout, 3, padding=1, bias=False)
        self.wq = LSQ(4, True, cout)

    def forward(self, x):
        return F.conv2d(x, self.wq(self.weight), None, 1, 1)


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        cin = 3
        for v in CFG:
            if v == "M":
                self.blocks.append(nn.MaxPool2d(2))
            else:
                self.blocks.append(
                    nn.ModuleDict(dict(conv=QConv(cin, v), bn=nn.BatchNorm2d(v), aq=LSQ(4, False)))
                )
                cin = v
        self.fc = nn.Linear(cin, 10)

    def forward(self, x, hook=None):
        li = 0
        for b in self.blocks:
            if isinstance(b, nn.MaxPool2d):
                x = b(x)
                continue
            if hook is not None:
                x = hook(li, b, x)  # analysis replaces conv+bn+relu+quant
            else:
                x = b["aq"](F.relu(b["bn"](b["conv"](x))))
            li += 1
        return self.fc(torch.flatten(x, 1))

    def set_quant(self, on):
        for m in self.modules():
            if isinstance(m, LSQ):
                m.enabled = on


def loaders(bs=128, workers=6):
    import torchvision, torchvision.transforms as T

    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tr = torchvision.datasets.CIFAR10(
        DATA,
        True,
        T.Compose([T.RandomCrop(32, 4), T.RandomHorizontalFlip(), T.ToTensor(), norm]),
        download=False,
    )
    te = torchvision.datasets.CIFAR10(DATA, False, T.Compose([T.ToTensor(), norm]), download=False)
    calib = torchvision.datasets.CIFAR10(DATA, True, T.Compose([T.ToTensor(), norm]), download=False)
    mk = lambda d, sh: torch.utils.data.DataLoader(d, bs, shuffle=sh, num_workers=workers)
    return mk(tr, True), mk(te, False), calib


def evaluate(model, loader):
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            ok += (model(x).argmax(1) == y).sum().item()
            n += len(y)
    return ok / n


def train(epochs, warm):
    torch.set_num_threads(os.cpu_count())
    tr, te, _ = loaders()
    model = VGG().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, epochs=epochs, steps_per_epoch=len(tr))
    for ep in range(epochs):
        model.set_quant(ep >= warm)  # short FP warm-up, then W4A4 QAT
        model.train()
        t0 = time.time()
        tot = ok = 0
        for x, y in tr:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += len(y)
            ok += (out.argmax(1) == y).sum().item()
        acc = evaluate(model, te) if (ep >= warm and (ep % 5 == 4 or ep == epochs - 1)) else float("nan")
        print(
            f"epoch {ep:3d} quant={ep >= warm} train {ok/tot:.4f} test {acc:.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, CKPT)
    print(f"final W4A4 test accuracy {evaluate(model, te):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warm", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.epochs, a.warm)
    else:
        m = VGG().to(DEV)
        m.load_state_dict(torch.load(CKPT, map_location=DEV))
        print(evaluate(m, loaders()[1]))
