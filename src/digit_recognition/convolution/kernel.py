from typing import Any


class Kernel():
    def __init__(self, schema: list[list[float]]) -> None:
        self.schema=schema

    def __eq__(self, other: Any):
        return self.schema == other.schema  # Assuming the kernel has an attribute called 'schema'
