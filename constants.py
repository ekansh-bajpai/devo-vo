import enum

class ScorerType(str, enum.Enum):
    POISSON = "poisson"
    TEMPORAL = "temporal"
    SNN = "snn"
    CNN = "cnn"


SCORER_TYPE = ScorerType.CNN
HAS_DESKTOP_ENV = False