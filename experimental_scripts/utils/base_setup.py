from typing import Iterable, Any
from re import compile

from pathlib import Path
import yaml


class Data:
    def __init__(self, *, name: str = 'Data', data: Any = None):
        self._name = name
        self._data = {}
        if data:
            for key, value in data.items():
                if isinstance(value, str) and value.startswith('__get_from_file:'):
                    self._data[key] = Data(data=BaseSetup.get_section_from_file(value.split(':', maxsplit=2)[-1]),
                                           name=key)
                else:
                    self._data[key] = Data(data=value, name=key) if isinstance(value, dict) else value

    def __repr__(self):
        return f'<{self._name}: {self._data}>'

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        raise AttributeError(f'{self._name}  has no "{key}" attribute')

    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]
        raise KeyError(f'{self._name} has no "{key}" key')

    def __contains__(self, key):
        return key in self._data

    def get(self, key: str, default: Any = None):
        return self._data.get(key, default)

    @property
    def name(self) -> str:
        return self._name

    def validate_required_fields(self, *fields):
        errors = [f'Field "{field}" is required' for field in fields if field not in self._data]
        if errors:
            assert False, '. '.join(errors)

    def validate_paths(self, *, exists: Iterable = (), create: Iterable = ()):
        for p in exists:
            path = self.path(key=p)
            assert path.is_dir(), f'Directory does not exist: {path}'
            self._data[p] = path.resolve()
        for p in create:
            path = self.path(key=p)
            path.mkdir(parents=True, exist_ok=True)
            assert path.is_dir(), f'Directory does not exist: {path}'
            self._data[p] = path.resolve()

    def path(self, *, key: str) -> Path:
        assert key in self._data, f'Key {key} not found in {self.name}'
        return Path(self[key])

    def compile_patterns(self, key: str, *, pattern_key: str, compiled_key: str):
        assert isinstance(self[key], list)
        for item in self[key]:
            if pattern_key in item:
                item[compiled_key] = compile(item[pattern_key])

    def set_value(self, *, key: str, value: Any):
        self._data[key] = value


class BaseSetup:
    _data: Data | None = None
    _path: Path | None = None

    DataType = Data

    def __init__(self):
        assert False

    @classmethod
    def init(cls, path: Path, *, name: str = 'setup') -> Data:
        assert cls._data is None, 'Cannot init BaseSetup more than once'
        assert path.is_file(), f'Setup file {path} not found'
        BaseSetup._path = path
        with open(path) as stream:
            cls._data = Data(data=yaml.load(stream, yaml.Loader), name=name)
        return cls._data

    @classmethod
    def get(cls) -> Data:
        assert cls._data is not None
        return cls._data

    @classmethod
    def get_section_from_file(cls, filename: str):
        assert cls._path is not None
        section_path = cls._path.parent / filename
        assert section_path.is_file(), f'Section file {section_path} not found'
        with open(section_path) as stream:
            return yaml.load(stream, yaml.Loader)


if __name__ == '__main__':
    setup = BaseSetup.init(Path(__file__).parent / 'test_setup.yaml')
    setup.validate_required_fields('video', 'basic_run')
    setup.video.validate_required_fields('recorded', 'processed', 'error')
    setup.video.validate_paths(exists=('recorded',), create=('processed', 'error'))
    assert 'video' in setup
    print(setup)
    print(setup.DataLocation)

