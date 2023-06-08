from typing import Iterable, Tuple, Callable, Union
from collections import namedtuple
from abc import ABC, abstractmethod
import re
import bz2
import csv
from pathlib import Path


ObjectTriple = namedtuple('ObjectTriple', 'sub pred obj')
LiteralTriple = namedtuple('LiteralTriple', 'sub pred obj')


class EdgelistReader(ABC):
    @abstractmethod
    def read(self, path: Path) -> Iterable[Tuple[str, str, str]]:
        """Read rows from a path. Returns (lhs, rel, rhs)."""
        pass


class NTriplesEdgelistReader(EdgelistReader):
    def read(self, path: Path) -> Iterable[Union[ObjectTriple, LiteralTriple]]:
        object_pattern = re.compile(rb'<(.+)> <(.+)> <(.+)> \.\s*\n')
        literal_pattern = re.compile(rb'<(.+)> <(.+)> "(.+)"(?:\^\^.*|@en.*)? \.\s*\n')
        with _get_open_fct(path)(path, "rb") as tf:
            for line_num, line in enumerate(tf, start=1):
                object_triple = object_pattern.match(line)
                if object_triple:
                    yield tuple([x.decode('utf-8') for x in object_triple.groups()])
                    continue
                literal_triple = literal_pattern.match(line)
                if literal_triple:
                    yield tuple([x.decode('utf-8') for x in literal_triple.groups()])
                    continue
                # TODO: log skipped line


class TSVEdgelistReader(EdgelistReader):
    def read(self, path: Path) -> Iterable[Union[ObjectTriple, LiteralTriple]]:
        with _get_open_fct(path)(path, newline='') as tf:
            for row in csv.reader(tf, delimiter='\t'):
                if len(row) < 3:
                    continue  # TODO: log skipped line
                yield tuple(row[:3])


def _get_open_fct(path: Path) -> Callable:
    return bz2.open if str(path).endswith('.bz2') else open


def get_reader_for_format(kg_format: str) -> EdgelistReader:
    if kg_format == 'tsv':
        return TSVEdgelistReader()
    if kg_format == 'nt':
        return NTriplesEdgelistReader()
    raise NotImplementedError(f'Reader for format "{kg_format}" not implemented.')
