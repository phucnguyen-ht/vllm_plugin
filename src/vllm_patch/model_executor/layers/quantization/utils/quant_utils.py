from typing import ClassVar, NamedTuple

class _GroupShape(NamedTuple):
    row: int
    col: int


class GroupShape(_GroupShape):
    """
    This class describes the quantization group shape.
    It includes static members for common shapes (per-tensor, per-token).
    """

    # Aliases for common quantization group shapes
    PER_TENSOR: ClassVar["GroupShape"]
    PER_TOKEN: ClassVar["GroupShape"]

    def is_per_tensor(self) -> bool:
        return self.row == -1 and self.col == -1

    def is_per_token(self) -> bool:
        return self.row == 1 and self.col == -1

    def is_per_group(self) -> bool:
        return self.row == 1 and self.col >= 1


GroupShape.PER_TENSOR = GroupShape(-1, -1)
GroupShape.PER_TOKEN = GroupShape(1, -1)