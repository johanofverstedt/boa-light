"""Label definitions of the Body Composition Analysis (BCA).

Ported from BOA ``body_composition_analysis`` (Apache-2.0, (c) IKIM / SHIP-AI, UK Essen).
"""

import enum


class BodyParts(enum.IntEnum):
    BACKGROUND = 0
    TORSO = 1
    HEAD = 2
    LEG_RIGHT = 3
    LEG_LEFT = 4
    ARM_RIGHT = 5
    ARM_LEFT = 6


class BodyRegion(enum.IntEnum):
    SUBCUTANEOUS_TISSUE = 1
    MUSCLE = 2
    ABDOMINAL_CAVITY = 3
    THORACIC_CAVITY = 4
    BONE = 5
    GLANDS = 6
    PERICARDIUM = 7
    BREAST_IMPLANT = 8
    MEDIASTINUM = 9
    BRAIN = 10
    NERVOUS_SYSTEM = 11


class HURange(enum.Enum):
    ALL = (-1000, 3000)
    ADIPOSE_TISSUE = (-190, -30)
    MUSCLE_TISSUE = (-29, 150)


class Tissue(enum.IntEnum):
    MUSCLE = 1
    BONE = 2
    SAT = 3
    VAT = 4
    IMAT = 5
    PAT = 6
    EAT = 7


TISSUE_DERIVATION_RULES = {
    Tissue.MUSCLE: (HURange.MUSCLE_TISSUE, BodyRegion.MUSCLE),
    Tissue.BONE: (HURange.ALL, BodyRegion.BONE),
    Tissue.SAT: (HURange.ADIPOSE_TISSUE, BodyRegion.SUBCUTANEOUS_TISSUE),
    Tissue.VAT: (HURange.ADIPOSE_TISSUE, BodyRegion.ABDOMINAL_CAVITY),
    Tissue.IMAT: (HURange.ADIPOSE_TISSUE, BodyRegion.MUSCLE),
    Tissue.PAT: (HURange.ADIPOSE_TISSUE, BodyRegion.MEDIASTINUM),
    Tissue.EAT: (HURange.ADIPOSE_TISSUE, BodyRegion.PERICARDIUM),
}
