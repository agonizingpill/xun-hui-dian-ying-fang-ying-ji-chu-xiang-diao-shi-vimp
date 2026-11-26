from enum import StrEnum


class BoothState(StrEnum):
    SURVEY = 'survey'
    POWERED = 'powered'
    RETEST = 'retest'
    CLEARED = 'cleared'


class LensState(StrEnum):
    STORED = 'stored'
    ON_LOAN = 'on_loan'


class LoanState(StrEnum):
    ACTIVE = 'active'
    RETURNED = 'returned'


class Artifact(StrEnum):
    CLEAN = 'clean'
    SOFT = 'soft_focus'
    SHIFT = 'color_shift'
    FLICKER = 'flicker'


BASELINE_MODEL_VERSION = 'ext-20261124-demo'
ANOMALY_ARTIFACTS = (Artifact.SOFT, Artifact.SHIFT, Artifact.FLICKER)
