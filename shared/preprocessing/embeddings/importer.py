from typing import List, Iterable, Tuple
import re
import bz2
from pathlib import Path
from torchbiggraph.config import ConfigSchema
from torchbiggraph.converters.importers import convert_input_data, EdgelistReader, TSVEdgelistReader


class NTriplesEdgelistReader(EdgelistReader):
    def read(self, path: Path) -> Iterable[Tuple[str, str, str]]:
        object_pattern = re.compile(rb'<(.+)> <(.+)> <(.+)> \.\s*\n')
        open_fct = bz2.open if str(path).endswith('.bz2') else open
        with open_fct(path, "rb") as tf:
            for line_num, line in enumerate(tf, start=1):
                object_triple = object_pattern.match(line)
                if not object_triple:
                    continue  # TODO: log skipped line
                sub, pred, obj = [x.decode('utf-8') for x in object_triple.groups()]
                yield sub, obj, pred, None


def convert_graph_data(kg_config: dict, embedding_config: ConfigSchema, input_files: List[Path]):
    convert_input_data(
        embedding_config.entities,
        embedding_config.relations,
        embedding_config.entity_path,
        embedding_config.edge_paths,
        input_files,
        _get_reader_for_format(kg_config['format']),
        dynamic_relations=embedding_config.dynamic_relations,
    )


def _get_reader_for_format(kg_format: str) -> EdgelistReader:
    if kg_format == 'tsv':
        return TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1)
    if kg_format == 'nt':
        return NTriplesEdgelistReader()
    raise NotImplementedError(f'Reader for format "{kg_format}" not implemented.')
