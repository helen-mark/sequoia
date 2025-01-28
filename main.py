from path_setup import SequoiaPath
from utils.parse_raw_data import check_and_parse
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
    dt1 = datetime.now()
    dt2 = datetime(2023, 1, 25)
    print(dt1.timestamp() - dt2.timestamp())

    check_and_parse(data_config, dataset_config)

    check_and_parse(data_config, dataset_config)


if __name__ == '__main__':
    main()