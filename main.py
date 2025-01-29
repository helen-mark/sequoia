from path_setup import SequoiaPath
from utils.sequoia_dataset import SequoiaDataset
from datetime import datetime
import yaml


def main():
    SequoiaPath()
    setup_path = 'data_config.yaml'
    dataset_config_path = 'dataset_config.yaml'
    with open(setup_path) as stream:
        data_config = yaml.load(stream, yaml.Loader)
        # for key, val in data_config.items():
        #     print(val)

    with open(dataset_config_path) as stream:
        dataset_config = yaml.load(stream, yaml.Loader)
        # print(dataset_config['snapshot_features']["common"])

    new_dataset = SequoiaDataset(data_config, dataset_config)
    new_dataset.check_and_parse()


if __name__ == '__main__':
    main()
