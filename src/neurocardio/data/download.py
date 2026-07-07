from pathlib import Path

import wfdb


def download_mitdb(dest: str = "data/mitdb") -> Path:
    """Download the MIT-BIH Arrhythmia Database from PhysioNet."""
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(out))
    return out


def download_ptbdb(dest: str = "data/ptbdb") -> Path:
    """Download the PTB Diagnostic ECG Database (extension dataset)."""
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("ptbdb", str(out))
    return out


def download_svdb(dest: str = "data/svdb") -> Path:
    """Download the MIT-BIH Supraventricular Arrhythmia Database (128 Hz).

    Same WFDB beat-annotation scheme as MIT-BIH; rich in SVEB. Used as an external
    cross-database test set (resample to 360 Hz before segmenting)."""
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("svdb", str(out))
    return out


def download_incartdb(dest: str = "data/incartdb") -> Path:
    """Download the St. Petersburg INCART 12-lead Arrhythmia Database (257 Hz).

    Same WFDB beat-annotation scheme; ~175k beats, rich in ventricular ectopy.
    Used as an external cross-database test set (resample to 360 Hz)."""
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("incartdb", str(out))
    return out


DATABASES = {
    "mitdb": download_mitdb,
    "ptbdb": download_ptbdb,
    "svdb": download_svdb,
    "incartdb": download_incartdb,
}


def download_db(slug: str, dest: str | None = None) -> Path:
    """Download a supported PhysioNet database by slug into dest (default data/<slug>)."""
    if slug not in DATABASES:
        raise ValueError(f"unknown database {slug!r} (known: {sorted(DATABASES)})")
    return DATABASES[slug](dest or f"data/{slug}")
