"""Make ``nnunetv2`` importable without torch for numpy-only parts (preprocessing tests)."""

import importlib.machinery
import sys
import types
from unittest import mock

STUBBED = ("torch", "SimpleITK", "blosc2", "tqdm", "dynamic_network_architectures", "timm", "einops",
           "batchgeneratorsv2", "threadpoolctl", "graphviz", "imagecodecs", "yacs", "seaborn", "matplotlib")


class _Loader:
    def create_module(self, spec):
        if spec.name == "torch":
            m = types.ModuleType("torch")
            m.__path__ = []

            class Tensor:  # real class so isinstance() works
                pass

            m.Tensor = Tensor
            fallback = mock.MagicMock()
            m.__getattr__ = lambda name: getattr(fallback, name)
            return m
        m = mock.MagicMock(name=spec.name)
        m.__path__ = []
        if spec.name == "dynamic_network_architectures.building_blocks.helper":
            # used by nnU-Net's old-plans conversion only for class names
            def cls(name, module):
                return type(name, (), {"__module__": module})

            m.convert_dim_to_conv_op = lambda dim: cls(f"Conv{dim}d", "torch.nn.modules.conv")
            m.get_matching_instancenorm = lambda **k: cls("InstanceNorm3d", "torch.nn.modules.instancenorm")
        return m

    def exec_module(self, module):
        pass


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in STUBBED:
            return importlib.machinery.ModuleSpec(fullname, _Loader(), is_package=True)
        return None


def install():
    sys.meta_path.insert(0, _Finder())
