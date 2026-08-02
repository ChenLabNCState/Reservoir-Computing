from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class SweepSpec:
    name: str
    values: Iterable
    label: str
    title: str
    save: bool