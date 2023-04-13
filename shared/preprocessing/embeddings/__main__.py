from pathlib import Path
import yaml
import attr
from torchbiggraph.config import add_to_sys_path, ConfigFileLoader
from importer import convert_graph_data
from torchbiggraph.converters import export_to_tsv
from torchbiggraph.train import train
from torchbiggraph.util import set_logging_verbosity, setup_logging, SubprocessInitializer


KG_DIR = Path('./kg')
EMBEDDINGS_DIR = KG_DIR / 'embeddings'
CHECKPOINTS_DIR = EMBEDDINGS_DIR / 'model'
ENTITY_EMBEDDING_FILE = EMBEDDINGS_DIR / 'entity_embeddings.tsv'
RELATION_TYPE_PARAMS_FILE = EMBEDDINGS_DIR / 'relation_types_parameters.tsv'


def make_embeddings():
    setup_logging()
    # load embedding config
    loader = ConfigFileLoader()
    config = loader.load_config('default_config.py')
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)
    # load and check KG config
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)['kg']
    input_edge_paths = [KG_DIR / filename for filename in kg_config['input_files']]
    # convert graph data
    config = attr.evolve(
        config,
        entity_path=str(EMBEDDINGS_DIR),
        edge_paths=[str(EMBEDDINGS_DIR / str(idx)) for idx in range(len(input_edge_paths))],
        checkpoint_path=str(CHECKPOINTS_DIR)
    )
    convert_graph_data(kg_config, config, input_edge_paths)
    # train model and embeddings
    train(config, subprocess_init=subprocess_init)
    # write embeddings
    with open(ENTITY_EMBEDDING_FILE, 'xt') as entities_tf, open(RELATION_TYPE_PARAMS_FILE, 'xt') as relation_types_tf:
        export_to_tsv.make_tsv(config, entities_tf, relation_types_tf)


if __name__ == "__main__":
    make_embeddings()
