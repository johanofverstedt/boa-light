"""Persistent, in-memory nnU-Net inference.

TotalSegmentator/BOA build a new ``nnUNetPredictor`` for every call, re-read the
checkpoint, start preprocessing/export process pools and pass the image through
gzipped temporary NIfTI files. Here every network is loaded once and kept for the
whole run, and images are handed over as numpy arrays.

To stay numerically identical to TotalSegmentator, the array given to nnU-Net is
exactly what nnU-Net's reader would have produced from TotalSegmentator's
temporary file: the voxel array in (z, y, x) order as float32, with the spacing
reversed, taken from the (float32) NIfTI header. This holds for both readers used
by these models (``SimpleITKIO`` for BCA, ``NibabelIOWithReorient`` for the
TotalSegmentator models), because the input is always in canonical (RAS) order.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import nibabel as nib
import numpy as np

from boa_light.models import NETWORKS
from boa_light.weights import model_folder

logger = logging.getLogger(__name__)


def resolve_device(device: str = "auto"):
    import torch

    if device in ("auto", None):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device in ("gpu",):
        device = "cuda"
    if device.startswith("gpu:"):
        device = "cuda:" + device.split(":", 1)[1]
    return torch.device(device)


class ModelPool:
    """Lazily loads nnU-Net networks and keeps them for reuse."""

    def __init__(self, device: str = "auto", weights_root: Path | None = None, keep_loaded: bool = True):
        import torch

        self.device = resolve_device(device)
        self.weights_root = weights_root
        self.keep_loaded = keep_loaded
        self._predictors: dict[tuple, object] = {}
        self._lock = threading.Lock()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------ #
    def _load(self, task_id: int, folds, step_size: float):
        import torch
        from batchgenerators.utilities.file_and_folder_operations import load_json
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        net = NETWORKS[task_id]
        folder = model_folder(net, self.weights_root)
        if folds is None:
            folds = sorted(
                int(p.parent.name.split("_")[-1])
                for p in folder.glob("fold_*/checkpoint_final.pth")
                if p.parent.name != "fold_all"
            )
        dataset_json = load_json(str(folder / "dataset.json"))
        plans_manager = PlansManager(load_json(str(folder / "plans.json")))
        parameters = []
        configuration_name = net.configuration
        mirroring_axes = None
        for i, f in enumerate(folds):
            ckpt = torch.load(
                folder / f"fold_{f}" / "checkpoint_final.pth", map_location="cpu", weights_only=False
            )
            if i == 0:
                configuration_name = ckpt["init_args"]["configuration"]
                mirroring_axes = ckpt.get("inference_allowed_mirroring_axes")
            parameters.append(ckpt["network_weights"])
        configuration_manager = plans_manager.get_configuration(configuration_name)
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        label_manager = plans_manager.get_label_manager(dataset_json)
        # The trainers used by these models only change training (epochs / mirroring
        # augmentation), so the default nnU-Net architecture builder applies. This
        # avoids having to register TotalSegmentator's/BOA's custom trainer classes.
        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            label_manager.num_segmentation_heads,
            allow_init=True,
            deep_supervision=False,
        )
        network.load_state_dict(parameters[0])
        predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=False,  # TotalSegmentator/BOA never use test-time mirroring
            perform_everything_on_device=self.device.type != "cpu",
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            network.to(self.device),
            plans_manager,
            configuration_manager,
            parameters,
            dataset_json,
            net.trainer,
            mirroring_axes,
        )
        if len(parameters) == 1:
            # Weights are already in the network; skip re-loading them on every call.
            predictor.list_of_parameters = [network.state_dict()]
        logger.info("Loaded network %s (folds %s) on %s", task_id, list(folds), self.device)
        return predictor

    def get(self, task_id: int, folds, step_size: float):
        key = (task_id, tuple(folds) if folds is not None else None, step_size)
        with self._lock:
            if key not in self._predictors:
                self._predictors[key] = self._load(task_id, folds, step_size)
            return self._predictors[key]

    def release(self, task_id: int) -> None:
        if self.keep_loaded:
            return
        with self._lock:
            for key in [k for k in self._predictors if k[0] == task_id]:
                del self._predictors[key]
        if self.device.type == "cuda":
            import torch

            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    def predict(self, task_id: int, img: nib.Nifti1Image, folds=(0,), step_size: float = 0.5) -> np.ndarray:
        """Segment a (canonical, resampled) NIfTI image; returns a uint8 array (x, y, z)."""
        predictor = self.get(task_id, folds, step_size)
        data = np.asanyarray(img.dataobj).astype(np.float32, copy=False)
        array = np.ascontiguousarray(data.transpose(2, 1, 0))[None]
        spacing = [float(s) for s in img.header.get_zooms()[:3]][::-1]
        seg = None
        if self.fast_export and len(predictor.list_of_parameters) == 1 and self.device.type != "cpu":
            try:
                seg = self._predict_argmax_on_device(predictor, array, spacing)
            except Exception:  # never fail because of the optimisation
                logger.warning("On-device export failed for task %s; using nnU-Net's export", task_id, exc_info=True)
                self.fast_export = False
        if seg is None:
            seg = predictor.predict_single_npy_array(array, {"spacing": spacing}, None, None, False)
        self.release(task_id)
        return np.asarray(seg).transpose(2, 1, 0).astype(np.uint8)

    fast_export = True

    @staticmethod
    def _predict_argmax_on_device(predictor, array: np.ndarray, spacing) -> np.ndarray | None:
        """Same result as ``predict_single_npy_array`` but the argmax runs on the GPU.

        nnU-Net moves the (half precision) logits of the whole volume to the CPU and
        takes a numpy argmax over the class axis, which for "total" at 1.5 mm is a
        multi-GB array. When no resampling of the logits is needed (always the case
        here, the input is already at the network spacing) the argmax can be taken
        on the device; ``torch.argmax`` and ``numpy.argmax`` both return the first
        maximum, so the result is identical. Returns None if the shortcut does not
        apply.
        """
        import torch
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy

        if predictor.label_manager.has_regions:
            return None
        ppa = PreprocessAdapterFromNpy(
            [array], [None], [{"spacing": spacing}], [None],
            predictor.plans_manager, predictor.dataset_json, predictor.configuration_manager,
            num_threads_in_multithreaded=1, verbose=False,
        )  # fmt: skip
        dct = next(ppa)
        props = dct["data_properties"]
        with torch.inference_mode():
            logits = predictor.predict_sliding_window_return_logits(dct["data"])
            if tuple(props["shape_after_cropping_and_before_resampling"]) != tuple(logits.shape[1:]):
                return None
            seg_cropped = torch.argmax(logits, 0).to(torch.uint8).cpu().numpy()
        del logits
        seg = np.zeros(props["shape_before_cropping"], dtype=np.uint8)
        seg = insert_crop_into_image(seg, seg_cropped, props["bbox_used_for_cropping"])
        return seg.transpose(predictor.plans_manager.transpose_backward)
