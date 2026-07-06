import argparse

import torch
from torch.utils.data import DataLoader

from neurocardio.config import load_config
from neurocardio.data.dataset import ECGBeatDataset, build_split
from neurocardio.data.segment import AAMI_CLASSES
from neurocardio.data.splits import get_split
from neurocardio.deploy.energy import spike_stats
from neurocardio.encoding.delta import delta_encode
from neurocardio.eval.evaluate import evaluate
from neurocardio.models.baselines import CNN1D, LSTMClassifier
from neurocardio.models.snn import SNNClassifier
from neurocardio.train.loop import set_seed, train


def _make_model(cfg):
    if cfg.model.kind == "snn":
        return SNNClassifier(
            in_features=2,
            hidden=cfg.model.hidden,
            n_classes=cfg.model.n_classes,
            beta=cfg.model.beta,
        )
    if cfg.model.kind == "cnn":
        return CNN1D(n_classes=cfg.model.n_classes)
    if cfg.model.kind == "lstm":
        return LSTMClassifier(n_classes=cfg.model.n_classes, hidden=cfg.model.hidden)
    raise ValueError(f"unknown model kind: {cfg.model.kind}")


def _make_dataset(cfg, record_ids):
    beats, labels = build_split(cfg, record_ids)
    if cfg.model.kind == "snn" and cfg.encoder.kind == "delta":
        thr = cfg.encoder.delta_threshold

        def transform(beat):
            return torch.from_numpy(delta_encode(beat, thr))

        return ECGBeatDataset(beats, labels, transform=transform)
    return ECGBeatDataset(beats, labels)


def cmd_download(args):
    from neurocardio.data.download import download_mitdb

    download_mitdb(args.dest)


def cmd_train(args):
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    train_ds = _make_dataset(cfg, get_split("train"))
    test_ds = _make_dataset(cfg, get_split("test"))
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size)
    model = _make_model(cfg)
    train(
        model,
        train_loader,
        test_loader,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        device=cfg.train.device,
    )
    result = evaluate(model, test_loader, classes=AAMI_CLASSES, device=cfg.train.device)
    print("Inter-patient (DS2) metrics:", result["metrics"])
    torch.save(model.state_dict(), args.out)


def cmd_evaluate(args):
    cfg = load_config(args.config)
    test_ds = _make_dataset(cfg, get_split("test"))
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size)
    model = _make_model(cfg)
    model.load_state_dict(torch.load(args.weights))
    result = evaluate(model, loader, classes=AAMI_CLASSES)
    print(result["metrics"])
    if cfg.model.kind == "snn":
        print("Energy proxy:", spike_stats(model, next(iter(loader))[0][:1]))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="neurocardio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download")
    p_dl.add_argument("--dest", default="data/mitdb")
    p_dl.set_defaults(func=cmd_download)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--config", default="configs/default.yaml")
    p_tr.add_argument("--out", default="runs/model.pt")
    p_tr.set_defaults(func=cmd_train)

    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--config", default="configs/default.yaml")
    p_ev.add_argument("--weights", required=True)
    p_ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
