# de Chazal et al. (2004) inter-patient split. Paced records (102, 104, 107,
# 217) are excluded per AAMI EC57. Keeping patients disjoint across train/test
# is the guardrail against beat-level leakage.
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]
PACED_RECORDS = ["102", "104", "107", "217"]


def get_split(name: str) -> list[str]:
    if name == "train":
        return DS1_RECORDS
    if name == "test":
        return DS2_RECORDS
    raise ValueError(f"unknown split: {name!r} (expected 'train' or 'test')")
