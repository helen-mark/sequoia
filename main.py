from path_setup import SequoiaPath
from utils.sequoia_dataset import SequoiaDataset
from datetime import datetime
import yaml


def main():
    SequoiaPath()
    setup_path = SequoiaPath.data_setup_file
    dataset_config_path = SequoiaPath.dataset_setup_file
    with open(setup_path) as stream:
        data_config = yaml.load(stream, yaml.Loader)

    with open(dataset_config_path) as stream:
        dataset_config = yaml.load(stream, yaml.Loader)
        # print(dataset_config['snapshot_features']["common"])

    new_dataset = SequoiaDataset(data_config, dataset_config)
    new_dataset.run()


if __name__ == '__main__':
    main()
