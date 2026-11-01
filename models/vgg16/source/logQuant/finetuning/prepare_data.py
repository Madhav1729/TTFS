"""Verify complete datasets and freeze train/development/test IDs for QAT."""

import argparse
import json
from pathlib import Path
import random

import numpy as np

from ..evaluate import read_tsv
from ..image_models.data import DATA_MD5, DATA_URL, digest, load_split


def stratified_cifar_split(labels, seed=42):
    """Hold out 50 of the 500 training images per class; never use test labels."""
    generator = random.Random(seed)
    validation = []
    for label in range(100):
        indices = np.flatnonzero(labels == label).tolist()
        if len(indices) != 500:
            raise ValueError("Expected 500 CIFAR-100 training images per class")
        generator.shuffle(indices)
        validation.extend(indices[:50])
    held_out = set(validation)
    train = [i for i in range(len(labels)) if i not in held_out]
    return train, sorted(validation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sst2-dir", type=Path, default=Path("logQuant/finetuning/data/SST-2"))
    parser.add_argument("--cifar-archive", type=Path,
                        default=Path("logQuant/finetuning/data/CIFAR100/cifar-100-python-mirror.tar.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("logQuant/finetuning/prepared/seed42"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if (args.output_dir / "data_manifest.json").exists():
        raise FileExistsError("Choose a fresh manifest output directory")
    sst_train, sst_dev = read_tsv(args.sst2_dir / "train.tsv"), read_tsv(args.sst2_dir / "dev.tsv")
    if len(sst_train) != 67349 or len(sst_dev) != 872:
        raise ValueError("Expected the complete SST-2 training/development splits")
    if digest(args.cifar_archive, "md5") != DATA_MD5:
        raise ValueError("CIFAR-100 archive differs from the official checksum")
    images, labels = load_split(args.cifar_archive, "train")
    del images
    train_ids, validation_ids = stratified_cifar_split(labels, args.seed)
    images, test_labels = load_split(args.cifar_archive, "test")
    if len(images) != 10000 or len(test_labels) != 10000:
        raise ValueError("Expected all 10,000 CIFAR-100 test images")
    del images
    manifest = dict(seed=args.seed, training_status="prepared; no fine-tuning run",
                    sst2=dict(source="https://huggingface.co/datasets/stanfordnlp/sst2",
                              directory=str(args.sst2_dir.resolve()), train_examples=len(sst_train),
                              development_examples=len(sst_dev),
                              train_sha256=digest(args.sst2_dir / "train.tsv"),
                              development_sha256=digest(args.sst2_dir / "dev.tsv"),
                              calibration_ids=random.Random(args.seed).sample(range(len(sst_train)), 128),
                              evaluation_note="Official validation is development data; official test labels are hidden."),
                    cifar100=dict(source=DATA_URL, archive=str(args.cifar_archive.resolve()),
                                  archive_md5=DATA_MD5, archive_sha256=digest(args.cifar_archive),
                                  train_examples=len(train_ids), development_examples=len(validation_ids),
                                  test_examples=len(test_labels), train_ids=train_ids,
                                  development_ids=validation_ids, test_ids=list(range(len(test_labels))),
                                  calibration_ids=random.Random(args.seed).sample(train_ids, 128),
                                  evaluation_note="Choose checkpoints using held-out training IDs; keep test separate."))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SST-2: 67,349 train / 872 development")
    print("CIFAR-100: 45,000 train / 5,000 development / 10,000 test")
    print("Saved", args.output_dir / "data_manifest.json")
    return manifest


if __name__ == "__main__":
    main()
