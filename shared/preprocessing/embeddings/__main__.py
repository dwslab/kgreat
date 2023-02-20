from pathlib import Path
import attr
from torchbiggraph.config import add_to_sys_path, ConfigFileLoader
from torchbiggraph.converters.importers import convert_input_data, TSVEdgelistReader
from torchbiggraph.converters import export_to_tsv
from torchbiggraph.train import train
from torchbiggraph.util import set_logging_verbosity, setup_logging, SubprocessInitializer

# TODO: adapt to arbitrary KG
KG_DIR = Path('./kg')
EMBEDDINGS_DIR = KG_DIR / 'embeddings'
CONFIG_FILE = 'default_config.py'
INPUT_FILE = 'dbpedia50k_train.tsv'
ENTITY_EMBEDDINGS_FILE = 'entity_embeddings.tsv'
RELATION_TYPE_PARAMS_FILE = 'relation_types_parameters.tsv'


def make_embeddings():
    setup_logging()

    # load config
    loader = ConfigFileLoader()
    config = loader.load_config(CONFIG_FILE)
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)

    config = attr.evolve(config,
                         entity_path=str(EMBEDDINGS_DIR),
                         edge_paths=[str(EMBEDDINGS_DIR)],
                         checkpoint_path=str(EMBEDDINGS_DIR / 'model')
                         )

    # convert graph data
    input_edge_paths = [KG_DIR / INPUT_FILE]
    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1),
        dynamic_relations=config.dynamic_relations,
    )

    # train model and embeddings
    train(config, subprocess_init=subprocess_init)

    # write embeddings
    with open(EMBEDDINGS_DIR / ENTITY_EMBEDDINGS_FILE, 'xt') as entities_tf, open(
            EMBEDDINGS_DIR / RELATION_TYPE_PARAMS_FILE, 'xt') as relation_types_tf:
        export_to_tsv.make_tsv(config, entities_tf, relation_types_tf)


if __name__ == "__main__":
    make_embeddings()
