from pathlib import Path

def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
