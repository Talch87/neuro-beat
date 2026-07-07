import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neurocardio.config import load_config
from neurocardio.data.dataset import ECGBeatDataset, build_external_split, build_split
from neurocardio.data.segment import AAMI_CLASSES
from neurocardio.data.splits import get_split
from neurocardio.deploy.energy import spike_stats
from neurocardio.encoding.delta import delta_encode
from neurocardio.eval.evaluate import evaluate
from neurocardio.models.baselines import CNN1D, LSTMClassifier
from neurocardio.models.snn import SNNClassifier
from neurocardio.train.loop import (
    class_weights_from_labels,
    resolve_device,
    set_seed,
    train,
)


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


def _dataset_from_beats(cfg, beats, labels):
    if cfg.model.kind == "snn" and cfg.encoder.kind == "delta":
        thr = cfg.encoder.delta_threshold

        def transform(beat):
            return torch.from_numpy(delta_encode(beat, thr))

        return ECGBeatDataset(beats, labels, transform=transform)
    return ECGBeatDataset(beats, labels)


def _make_dataset(cfg, record_ids):
    beats, labels = build_split(cfg, record_ids)
    return _dataset_from_beats(cfg, beats, labels)


def cmd_download(args):
    from neurocardio.data.download import download_db

    out = download_db(args.db, args.dest)
    print(f"downloaded {args.db} -> {out}")


def cmd_crossdb(args):
    """Evaluate trained weights on an external WFDB database (resampled to config fs)."""
    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    beats, labels = build_external_split(cfg, args.data_dir, lead_index=args.lead)
    loader = DataLoader(_dataset_from_beats(cfg, beats, labels), batch_size=cfg.train.batch_size)
    model = _make_model(cfg)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    result = evaluate(model, loader, classes=AAMI_CLASSES, device=device)
    print(f"External DB {args.data_dir}: {len(labels)} beats, lead={args.lead or cfg.data.lead_index}")
    print(result["metrics"])


def cmd_train(args):
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    train_ds = _make_dataset(cfg, get_split("train"))
    test_ds = _make_dataset(cfg, get_split("test"))
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size)
    model = _make_model(cfg)
    class_weights = None
    if cfg.train.class_weight == "balanced":
        class_weights = class_weights_from_labels(train_ds.labels, cfg.model.n_classes)
    print(f"device={device}  class_weight={cfg.train.class_weight}  batch={cfg.train.batch_size}")
    train(
        model,
        train_loader,
        test_loader,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        device=device,
        class_weights=class_weights,
    )
    result = evaluate(model, test_loader, classes=AAMI_CLASSES, device=device)
    print("Inter-patient (DS2) metrics:", result["metrics"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)


def cmd_evaluate(args):
    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    test_ds = _make_dataset(cfg, get_split("test"))
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size)
    model = _make_model(cfg)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    result = evaluate(model, loader, classes=AAMI_CLASSES, device=device)
    print(result["metrics"])
    if cfg.model.kind == "snn":
        model.to("cpu")  # energy proxy is a device-agnostic count
        print("Energy proxy:", spike_stats(model, next(iter(loader))[0][:1]))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="neurocardio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download")
    p_dl.add_argument("--db", default="mitdb", help="mitdb | svdb | incartdb | ptbdb")
    p_dl.add_argument("--dest", default=None, help="default data/<db>")
    p_dl.set_defaults(func=cmd_download)

    p_cx = sub.add_parser("crossdb", help="evaluate weights on an external WFDB database")
    p_cx.add_argument("--config", default="configs/default.yaml")
    p_cx.add_argument("--weights", required=True)
    p_cx.add_argument("--data-dir", required=True, help="downloaded external DB directory")
    p_cx.add_argument("--lead", type=int, default=None, help="lead index (default: config)")
    p_cx.set_defaults(func=cmd_crossdb)

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
