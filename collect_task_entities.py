import yaml
import pandas as pd
from pathlib import Path
from shared.dm.utils.dataset import load_dataset


def write_entities_files():
    """Instantiate datasets to collect all entities used in a task; then write the respective `entities.tsv` file."""
    for path_to_dataset in Path('dataset').glob('*'):
        path_to_config = path_to_dataset / 'config.yaml'
        if not path_to_config.is_file():
            continue
        with open(path_to_config) as f:
            dataset_config = yaml.safe_load(f)
        filepaths = [fp for fp in dataset_config if fp.endswith('_file')]
        for filepath in filepaths:
            dataset_config[filepath] = str(path_to_dataset / dataset_config[filepath])
        dataset = load_dataset(dataset_config, {}, pd.DataFrame())
        dataset.get_entities().to_csv(path_to_dataset / 'entities.tsv', sep='\t', index=False)


if __name__ == '__main__':
    write_entities_files()
