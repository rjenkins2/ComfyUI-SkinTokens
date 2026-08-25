from dataclasses import dataclass
try:
    import torch
    from torch import Tensor
except ImportError:
    Tensor = any # Fallback for environments without torch
from typing import Dict, Optional, List, Tuple

import io
import os
import sys
from ..rig_package.info.asset import Asset
try:
    from ..model.tokenrig import TokenRig
except ImportError:
    TokenRig = any

PORT = 59875
SERVER = f"http://localhost:{PORT}"
TMP_CKPT_DIR = "./tmp_ckpt"

BPY_PORT = 59876
BPY_SERVER = f"http://localhost:{BPY_PORT}"

@dataclass
class TensorPacket:
    """make sure stays on cpu"""
    validate: bool=False
    know_skeleton: bool=False
    learned_mesh_cond: Optional[Tensor]=None
    cond_latents: Optional[Tensor]=None
    mesh_cond: Optional[Tensor]=None
    vertices: Optional[Tensor]=None
    assets: Optional[List[Asset]]=None
    output_ids: Optional[Tensor]=None
    start_embed_list: Optional[List[Tensor]]=None
    start_tokens_list: Optional[List[List[int]]]=None

    def to_device(self, device):
        if self.learned_mesh_cond is not None:
            self.learned_mesh_cond = self.learned_mesh_cond.to(device)
        if self.cond_latents is not None:
            self.cond_latents = self.cond_latents.to(device)
        if self.mesh_cond is not None:
            self.mesh_cond = self.mesh_cond.to(device)
        if self.vertices is not None:
            self.vertices = self.vertices.to(device)
        if self.output_ids is not None:
            self.output_ids = self.output_ids.to(device)
        if self.start_embed_list is not None:
            self.start_embed_list = [x.to(device) for x in self.start_embed_list]

    @property
    def B(self):
        assert self.learned_mesh_cond is not None
        return self.learned_mesh_cond.shape[0]

    def to_bytes(self):
        return object_to_bytes(self)

    @classmethod
    def from_bytes(cls, bytes) -> 'TensorPacket':
        return bytes_to_object(bytes)

# Fix for object_to_bytes and bytes_to_object with relative paths.

import pickle

# Dotted name of the package this module lives in.
#   ComfyUI:  "ComfyUI-SkinTokens.src"
#   Blender:  "src"
# When the two are equal the remap is a no-op, so the same code runs in both
# processes with no branching.
_LOCAL = __name__.rsplit(".", 2)[0]
_WIRE = "src"

_classes = None

def _wire_classes():
    """Build (Pickler, Unpickler) subclasses that rewrite _LOCAL <-> _WIRE."""
    global _classes
    if _classes is not None:
        return _classes

    import dill
    from types import FunctionType
    from dill._dill import (
        MetaCatchingDict,
        save_type as _stock_save_type,
        save_function as _stock_save_function,
    )

    def _emit(pickler, obj, mod):
        pickler.save(_WIRE + mod[len(_LOCAL):])
        pickler.save(getattr(obj, "__qualname__", None) or obj.__name__)
        pickler.write(pickle.STACK_GLOBAL)
        pickler.memoize(obj)

    def _local(obj):
        mod = getattr(obj, "__module__", "") or ""
        return mod if (mod == _LOCAL or mod.startswith(_LOCAL + ".")) else None

    def _save_type(pickler, obj, *args, **kwargs):
        mod = _local(obj)
        if mod is not None:
            return _emit(pickler, obj, mod)
        return _stock_save_type(pickler, obj, *args, **kwargs)

    def _save_function(pickler, obj, *args, **kwargs):
        mod = _local(obj)
        if mod is not None:
            return _emit(pickler, obj, mod)
        return _stock_save_function(pickler, obj, *args, **kwargs)

    class WirePickler(dill.Pickler):
        # dill dispatches type/function saving through this table, and its
        # save_type calls StockPickler.save_global unbound, so overriding
        # save_global on the subclass would not be reached.
        dispatch = MetaCatchingDict(dill.Pickler.dispatch)

    WirePickler.dispatch[type] = _save_type
    WirePickler.dispatch[FunctionType] = _save_function

    class WireUnpickler(dill.Unpickler):
        def find_class(self, module, name):
            if module == _WIRE or module.startswith(_WIRE + "."):
                module = _LOCAL + module[len(_WIRE):]
            return super().find_class(module, name)

    _classes = (WirePickler, WireUnpickler)
    return _classes


def object_to_bytes(t):
    import dill
    if _LOCAL == _WIRE:
        return dill.dumps(t)
    pickler_cls, _ = _wire_classes()
    buf = io.BytesIO()
    # STACK_GLOBAL needs protocol 4 or newer
    pickler_cls(buf, protocol=max(4, pickle.DEFAULT_PROTOCOL)).dump(t)
    return buf.getvalue()


def bytes_to_object(b, map_location=None):
    import dill
    if _LOCAL == _WIRE:
        return dill.loads(b)
    _, unpickler_cls = _wire_classes()
    return unpickler_cls(io.BytesIO(b)).load()


def get_model(
    ckpt_path: str,
    hf_path: Optional[str]=None,
    device='cuda',
) -> TokenRig:
    model = TokenRig.load_from_system_checkpoint(checkpoint_path=ckpt_path)
    if hf_path is not None:
        from transformers import AutoModel
        a = AutoModel.from_pretrained(
            hf_path,
            local_files_only=True,
            _attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if 'torch' in sys.modules else None,
        )
        model.transformer.model.load_state_dict(a.state_dict())

    model = model.to(device)
    return model
