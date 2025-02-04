import json
import re
from shutil import rmtree

from pathlib import Path
import yaml


def load_data(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Cannot read setup from {path.as_posix()}')
    with open(path) as stream:
        if path.name.lower().endswith('.yaml') or path.name.lower().endswith('.yml'):
            return yaml.load(stream, yaml.Loader)
        elif path.name.lower().endswith('.json'):
            return json.load(stream)
    raise ValueError(f'Setup file format {path.name} not supported, use yaml/yml or json format')


class Setup:
    REQUIRED_FIELDS = ('DataLocation', 'BasicRun', 'Logging')

    def __init__(self, filepath: Path):
        if isinstance(filepath, str):
            filepath = Path(filepath)

        self._data = Setup.read_setup_file(filepath)
        self.parent_path = Path(__file__).parent.parent.parent.resolve()
        self.source_file = filepath.resolve()
        self.validate_required_fields(*Setup.REQUIRED_FIELDS)
        assert self.parent_path.is_dir()

    @staticmethod
    def read_setup_file(filepath: Path):
        if not filepath.is_file():
            raise FileNotFoundError(f'No setup file found: {filepath.resolve()}')
        with open(filepath) as stream:
            if filepath.name.lower().endswith('.yaml') or filepath.name.lower().endswith('.yml'):
                return yaml.load(stream, yaml.Loader)
            elif filepath.name.lower().endswith('.json'):
                return json.load(stream)
        raise ValueError(f'Setup file format {filepath.name} not supported, use yaml/yml or json format')

    def validate_required_fields(self, *fields):
        for field in fields:
            assert field in self._data, f'Required field {field} not found in setup file'

    def validate_required_paths(self, *names):
        data_location = self._data['DataLocation']
        for key in names:
            assert key in data_location, f'Field {key} is required in DataLocation section'
            path = Path(data_location[key])
            if not path.is_absolute():
                path = self.parent_path / path
            assert path.is_dir(), f'{path} shall be existing folder'
            data_location[key] = path.resolve()

    def create_output_dirs(self, *names):
        data_location = self._data['DataLocation']
        for key in names:
            assert key in data_location, f'Field {key} is required in DataLocation section'
            path = Path(data_location[key])
            if not path.is_absolute():
                path = self.parent_path / path
            path.mkdir(exist_ok=True)
            assert path.is_dir(), f'{path} shall be created'
            data_location[key] = path
        return self

    def compile_patterns(self):
        if 'TimeStamps' in self._data and isinstance(self._data['TimeStamps'], list):
            timestamps = self._data['TimeStamps']
            for pat in timestamps:
                if 'pattern' in pat:
                    pat['compiled'] = re.compile(pat['pattern'])

    def __getattr__(self, item):
        return self._data[item]

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __str__(self):
        s = f'Setup read from {self.source_file}\n'
        for key, value in self._data.items():
            s += f'{key}:\n'
            if isinstance(value, dict):
                for k, v in value.items():
                    s += f'\t{k}: {v}\n'
            elif isinstance(value, list):
                s += f'\t[{", ".join(str(_) for _ in value)}]\n'
            else:
                s += f'\t{str(value)}\n'
        return s

    def _validate_paths(self):
        assert 'DataLocation' in self._data
        data_location = self._data['DataLocation']
        assert 'PathVideo' in data_location and 'ProcessedVideo' in data_location and 'Intermediary' in data_location
        path = Path(__file__).parent.parent.parent.resolve()
        data_location['PathVideo']      = Setup._absolute_or_relative(path, Path(data_location['PathVideo']))
        data_location['ProcessedVideo'] = Setup._absolute_or_relative(path, Path(data_location['ProcessedVideo']))
        data_location['Intermediary']   = Setup._absolute_or_relative(path, Path(data_location['Intermediary']))

    @staticmethod
    def _absolute_or_relative(poultry_path: Path, path: Path):
        ret = path if path.is_absolute() else poultry_path / path
        assert ret.is_dir()
        return ret.resolve()

def __test(name):
    here = Path(__file__).parent
    setup_file = here / name
    test_setup = {
        'DataLocation': {'source': 'shared/utils/test_source', 'output': 'shared/utils/test_output'},
        'BasicRun': None,
        'Logging': None
    }
    with open(setup_file, 'w') as s:
        yaml.dump(test_setup, s)

    try:
        Setup(here / f'_{name}')
    except FileNotFoundError as e:
        pass

    test_source = here / 'test_source'
    test_source.mkdir(exist_ok=True)
    setup = Setup(here / name)
    setup.validate_required_paths('source')
    setup.create_output_dirs('output')
    assert setup.BasicRun is None
    assert setup.Logging is None
    assert setup.DataLocation['source'].is_dir()
    assert setup.DataLocation['output'].is_dir()

    TestSetup(setup_file)

    for d in ('source', 'output'):
        rmtree(setup.DataLocation[d])
    setup_file.unlink()

    print('Passed')


def load_setup(filename):
    return Setup(filename)


class TestSetup(Setup):
    def __init__(self, filepath: Path):
        super().__init__(filepath)
        self.validate_required_paths('source')
        self.create_output_dirs('output')


if __name__ == '__main__':
    __test('setup.yaml')
