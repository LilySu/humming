from enum import Enum


class QuantizationPhase(str, Enum):
    Fused = "fused"
    CollectAbsmax = "collect_absmax"
    Quantize = "quantize"


class ActivationType(str, Enum):
    None_ = "none"
    Unary = "unary"
    BinarySplit = "binary_split"
    BinaryInterleaved = "binary_interleaved"

    @property
    def cpp_name(self) -> str:
        return self.name.removesuffix("_")


class LayoutType(str, Enum):
    Normal = "normal"
    Permute = "permute"
    GroupedMask = "grouped_mask"
    Scatter = "scatter"


class GroupScaleLayout(str, Enum):
    RowMajor = "row_major"
    MMajor = "m_major"
    MxPacked = "mx_packed"
