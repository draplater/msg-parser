from collections import namedtuple


class EdgeToEmpty(namedtuple("_", "head id position")): # type: (int, int, int)
    pass
