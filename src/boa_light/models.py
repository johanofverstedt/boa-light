"""Declarative table of the networks and tasks BOA_light runs.

Every entry reproduces the parameters BOA (TotalSegmentator v2.12.0 semantics) uses
for the corresponding model. Only openly licensed (Apache-2.0) models are included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Networks (nnU-Net model folders)
# --------------------------------------------------------------------------- #

TS_URL = "https://github.com/wasserth/TotalSegmentator/releases/download"
BOA_URL = "https://github.com/UMEssen/Body-and-Organ-Analysis/releases/download"


@dataclass(frozen=True)
class Network:
    task_id: int
    folder: str  # dataset folder name inside the weights directory
    url: str
    trainer: str
    plans: str = "nnUNetPlans"
    configuration: str = "3d_fullres"


NETWORKS: dict[int, Network] = {
    n.task_id: n
    for n in [
        # TotalSegmentator "total" (1.5 mm, five part models)
        Network(291, "Dataset291_TotalSegmentator_part1_organs_1559subj",
                f"{TS_URL}/v2.0.0-weights/Dataset291_TotalSegmentator_part1_organs_1559subj.zip",
                "nnUNetTrainerNoMirroring"),
        Network(292, "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
                f"{TS_URL}/v2.0.0-weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip",
                "nnUNetTrainerNoMirroring"),
        Network(293, "Dataset293_TotalSegmentator_part3_cardiac_1559subj",
                f"{TS_URL}/v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip",
                "nnUNetTrainerNoMirroring"),
        Network(294, "Dataset294_TotalSegmentator_part4_muscles_1559subj",
                f"{TS_URL}/v2.0.0-weights/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip",
                "nnUNetTrainerNoMirroring"),
        Network(295, "Dataset295_TotalSegmentator_part5_ribs_1559subj",
                f"{TS_URL}/v2.0.0-weights/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip",
                "nnUNetTrainerNoMirroring"),
        # "total" fast (3 mm) and the 6 mm model used for rough crop masks
        Network(297, "Dataset297_TotalSegmentator_total_3mm_1559subj",
                f"{TS_URL}/v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip",
                "nnUNetTrainer_4000epochs_NoMirroring"),
        Network(298, "Dataset298_TotalSegmentator_total_6mm_1559subj",
                f"{TS_URL}/v2.0.0-weights/Dataset298_TotalSegmentator_total_6mm_1559subj.zip",
                "nnUNetTrainer_4000epochs_NoMirroring"),
        # Open subtasks
        Network(258, "Dataset258_lung_vessels_248subj",
                f"{TS_URL}/v2.0.0-weights/Dataset258_lung_vessels_248subj.zip", "nnUNetTrainer"),
        Network(150, "Dataset150_icb_v0",
                f"{TS_URL}/v2.0.0-weights/Dataset150_icb_v0.zip", "nnUNetTrainer"),
        Network(260, "Dataset260_hip_implant_71subj",
                f"{TS_URL}/v2.0.0-weights/Dataset260_hip_implant_71subj.zip", "nnUNetTrainer"),
        Network(315, "Dataset315_thoraxCT",
                f"{TS_URL}/v2.0.0-weights/Dataset315_thoraxCT.zip", "nnUNetTrainer"),
        Network(8, "Dataset008_HepaticVessel",
                f"{TS_URL}/v2.4.0-weights/Dataset008_HepaticVessel.zip", "nnUNetTrainer"),
        Network(570, "Dataset570_ct_liver_segments",
                f"{TS_URL}/v2.5.0-weights/Dataset570_ct_liver_segments.zip", "nnUNetTrainerNoMirroring"),
        # Body Composition Analysis (BOA)
        Network(542, "Dataset542_BCA_inference",
                f"{BOA_URL}/v1.0.0-weights/Dataset542_BCA_inference.zip", "nnUNetTrainerNoMirroring"),
        Network(543, "Dataset543_BCA_body_parts",
                f"{BOA_URL}/v1.0.0-weights/Dataset543_BCA_body_parts.zip",
                "nnUNetTrainer_1500epochs_NoMirroring"),
    ]
}  # fmt: skip

# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

LUNG_LOBES = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)


@dataclass(frozen=True)
class Task:
    """One segmentation output (one ``<name>.nii.gz``)."""

    name: str
    label_map_key: str  # key into class_maps.class_map
    task_ids: tuple[int, ...]  # >1 -> TotalSegmentator multi-model (labels are merged)
    resample: tuple[float, float, float] | None = None
    resample_only_thickness: bool = False  # BCA: only resample the slice thickness
    folds: tuple[int, ...] | None = (0,)  # None -> all available folds
    crop: tuple[str, ...] | None = None  # crop to these "total" ROIs (+20 mm)
    step_size: float = 0.5
    kind: str = "ts"  # "ts" | "bca"
    description: str = ""
    deps: tuple[str, ...] = field(default=())

    @property
    def multimodel(self) -> bool:
        return len(self.task_ids) > 1


# Rough 6 mm total segmentation, used by TotalSegmentator to crop subtasks.
# BOA re-computes it for every cropped task; BOA_light computes it once.
ROUGH = Task("_rough_total_6mm", "total", (298,), resample=(6.0, 6.0, 6.0))

TASKS: dict[str, Task] = {
    t.name: t
    for t in [
        Task("total", "total", (291, 292, 293, 294, 295), resample=(1.5, 1.5, 1.5), step_size=0.8,
             description="TotalSegmentator, 117 structures (1.5 mm)"),
        Task("lung_vessels", "lung_vessels", (258,), crop=LUNG_LOBES,
             description="Lung vessels and trachea/bronchia"),
        Task("cerebral_bleed", "cerebral_bleed", (150,), crop=("brain",),
             description="Intracerebral hemorrhage"),
        Task("hip_implant", "hip_implant", (260,),
             crop=("femur_left", "femur_right", "hip_left", "hip_right"), description="Hip implant"),
        Task("liver_vessels", "liver_vessels", (8,), crop=("liver",),
             description="Liver vessels and liver tumor"),
        Task("pleural_pericard_effusion", "pleural_pericard_effusion", (315,), folds=None,
             crop=LUNG_LOBES, description="Pleural and pericardial effusion"),
        Task("liver_segments", "liver_segments", (570,),
             resample=(0.8046879768371582, 0.8046879768371582, 1.5), crop=("liver",),
             description="Couinaud liver segments"),
        Task("body_parts", "body_parts", (543,), resample=(5.0, 5.0, 5.0), resample_only_thickness=True,
             folds=(0, 1, 2, 3, 4), kind="bca", description="BCA body parts (torso/head/limbs)"),
        Task("body_regions", "body_regions", (542,), resample=(5.0, 5.0, 5.0),
             resample_only_thickness=True, folds=(0, 1, 2, 3, 4), kind="bca",
             description="BCA body regions"),
    ]
}  # fmt: skip

# "total" with --fast-total (single 3 mm model)
TOTAL_FAST = Task("total", "total", (297,), resample=(3.0, 3.0, 3.0),
                  description="TotalSegmentator fast (3 mm)")  # fmt: skip

# User facing model names. "bca" expands to total + body_parts + body_regions + tissues.
SELECTABLE = ["total", "bca", "body_parts", "body_regions"] + [
    n for n, t in TASKS.items() if t.crop is not None
]
ALL_DEFAULT = [
    "total",
    "bca",
    "lung_vessels",
    "cerebral_bleed",
    "hip_implant",
    "liver_vessels",
    "pleural_pericard_effusion",
]  # == BOA's "all" without the licensed heartchambers_highres
CROP_TASKS = [n for n, t in TASKS.items() if t.crop is not None]


def resolve_models(spec: str | list[str] | None, bca_skip_body_parts: bool = False) -> tuple[list[str], bool]:
    """Turn a "+"-separated model string into (ordered tasks to run, run_bca).

    With ``bca_skip_body_parts`` the BCA does not add the ``body_parts`` model (it is
    only needed for the "no extremities" measurements and the L3 body axes); it is
    still run if requested explicitly.
    """
    if spec is None or spec == "all" or spec == ["all"]:
        names = list(ALL_DEFAULT)
    else:
        if isinstance(spec, str):
            spec = spec.split("+")
        names = [s.strip().replace("-", "_") for s in spec if s.strip()]
        if "all" in names:
            names = list(ALL_DEFAULT) + [n for n in names if n != "all"]
    unknown = sorted(set(names) - set(SELECTABLE))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Available: {', '.join(SELECTABLE)}")
    tasks: list[str] = []
    run_bca = "bca" in names
    if run_bca:
        names = ["total", *([] if bca_skip_body_parts else ["body_parts"]), "body_regions", *names]
    for n in ["total", *CROP_TASKS, "body_parts", "body_regions"]:
        if n in names and n not in tasks:
            tasks.append(n)
    return tasks, run_bca


def networks_for(tasks: list[str], fast_total: bool = False, crop_source: str = "rough") -> list[int]:
    ids: list[int] = []
    for name in tasks:
        task = TOTAL_FAST if (name == "total" and fast_total) else TASKS[name]
        ids.extend(task.task_ids)
    needs_rough = any(TASKS[n].crop for n in tasks) and not (crop_source == "total" and "total" in tasks)
    if needs_rough:
        ids.extend(ROUGH.task_ids)
    return list(dict.fromkeys(ids))
