from .base  import DVSLineAlgorithm
from .hough import PaperHoughLineAlgorithm, HoughLineParams, HOUGH_PRESETS
from .sam   import SamLineAlgorithm,        SamLineParams,   SAM_PRESETS
from src.shared import Spec
from .snn import SNN_regression, SNNLineParams, SNN_PRESETS

LINE_ALGO_REGISTRY = {
    "hough": Spec(
        PaperHoughLineAlgorithm,
        HoughLineParams,
        HOUGH_PRESETS,
    ),
    "sam": Spec(
        SamLineAlgorithm,
        SamLineParams,
        SAM_PRESETS,
    ),
    "snn": Spec(
        SNN_regression,
        SNNLineParams,
        SNN_PRESETS,
    )
}
