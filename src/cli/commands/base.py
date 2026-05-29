from abc import ABC, abstractmethod
from pathlib import Path

class Command(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def execute(self):
        pass

    def _get_files(self):
        raw = self.args.path
        paths = raw if isinstance(raw, list) else [raw]
        files = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                files.extend(path.rglob("opti_*.txt"))
            elif path.is_file():
                files.append(path)
        return sorted(set(files))