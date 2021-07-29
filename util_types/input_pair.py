from typing import Tuple
from collections import namedtuple

GanPair = namedtuple("GanPair", ("builder", "discriminator"))

CycleInput = Tuple[GanPair, GanPair]
