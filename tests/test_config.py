from pathlib import Path
from neurocardio.config import Config, load_config


def test_defaults():
    cfg = Config()
    assert cfg.data.fs == 360
    assert cfg.data.window_before + cfg.data.window_after == 256
    assert cfg.model.n_classes == 5
    assert cfg.encoder.kind == "delta"


def test_yaml_override_preserves_untouched_defaults(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  hidden: 64\ntrain:\n  epochs: 3\n")
    cfg = load_config(p)
    assert cfg.model.hidden == 64          # overridden
    assert cfg.train.epochs == 3           # overridden
    assert cfg.model.beta == 0.9           # default preserved
    assert cfg.data.fs == 360              # default preserved
