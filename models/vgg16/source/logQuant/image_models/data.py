"""Verified official CIFAR-100 archive and the checkpoint author's normalization."""

import hashlib
import pickle
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen

import numpy as np
import torch


CHECKPOINT_URL = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt"
CHECKPOINT_SHA256 = "7d8c4031aee657cfa20039610394e2381bc6d907e9b9a35190d51bceb172d30b"
DATA_URL = "https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz"
DATA_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
MODEL_COMMIT = "786c16252c0fc58ee9adac063f8337cc4a7a497a"
PREPROCESS_COMMIT = "610e6523ab462351700aaf629e6c739a9a30e903"
MEAN = (0.5070, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2761)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    partial.replace(path)


def load_split(path, split):
    if split not in ("train", "test"):
        raise ValueError("CIFAR-100 split must be train or test")
    # Read the exact member instead of extracting arbitrary archive paths.
    with tarfile.open(path, "r:gz") as archive:
        with archive.extractfile("cifar-100-python/" + split) as stream:
            batch = pickle.load(stream, encoding="bytes")
    images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[b"fine_labels"], dtype=np.int64)
    if len(images) != len(labels) or not ((labels >= 0) & (labels < 100)).all():
        raise ValueError("Invalid CIFAR-100 fine labels")
    return images, labels


def preprocess(images, device):
    value = torch.from_numpy(images).to(device=device, dtype=torch.float32) / 255
    mean = value.new_tensor(MEAN).view(1, 3, 1, 1)
    std = value.new_tensor(STD).view(1, 3, 1, 1)
    return (value - mean) / std
