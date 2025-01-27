import sys
from pathlib import Path


class SequoiaPath:
    sequoia: Path = Path(__file__).resolve().parent
    python: str = sys.executable

    utils: Path = sequoia / 'utils'
    data_setup_file: Path = sequoia / 'data_config.yaml'
    dataset_setup_file: Path = sequoia / 'dataset_config.yaml'

    _initialized: bool = False

    def __init__(self):
        if not self._initialized:
            if str(self.sequoia) not in sys.path:
                sys.path.append(str(self.sequoia))
            SequoiaPath._initialized = True
        assert self.data_setup_file.is_file(), f"File {self.data_setup_file} doesn't exist"

    @classmethod
    def initialized(cls) -> bool:
        return cls._initialized


if __name__ == '__main__':
    print(SequoiaPath.initialized())
    print(SequoiaPath.sequoia, SequoiaPath.python)
    for item in sys.path:
        print('\t', item)
    SequoiaPath()
    print(SequoiaPath.initialized())
    for item in sys.path:
        print('\t', item)
