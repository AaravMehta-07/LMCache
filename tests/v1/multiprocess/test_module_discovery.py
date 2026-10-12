# SPDX-License-Identifier: Apache-2.0
"""Tests for engine-module discovery and construction."""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import (
    ModuleBuildContext,
    discover_modules,
    order_modules,
)
from lmcache.v1.multiprocess.modules import engine_driven_transfer as ed_mod
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as ld_mod
from lmcache.v1.multiprocess.modules import lookup as lookup_mod
from lmcache.v1.multiprocess.modules import management as management_mod
from lmcache.v1.multiprocess.modules import p2p_controller as p2p_mod
from lmcache.v1.multiprocess.modules.blend import module as blend_mod
from lmcache.v1.multiprocess.modules.experimental import qstore as qstore_mod

EXPECTED_MODULE_NAMES = {
    "lookup",
    "p2p_controller",
    "management",
    "lmcache_driven_transfer",
    "engine_driven_transfer",
    "qstore",
    "blend",
}


@pytest.fixture
def stub_constructors(monkeypatch):
    def _stub(self, *args, **kwargs):
        return None

    for module, class_name in [
        (lookup_mod, "LookupModule"),
        (p2p_mod, "P2PController"),
        (ld_mod, "LMCacheDrivenTransferModule"),
        (ed_mod, "EngineDrivenTransferModule"),
        (qstore_mod, "QStoreModule"),
        (blend_mod, "BlendModule"),
        (management_mod, "ManagementModule"),
    ]:
        monkeypatch.setattr(getattr(module, class_name), "__init__", _stub)


def _build(**kwargs):
    return server_mod._build_modules(
        MagicMock(name="ctx"),
        MPServerConfig(**kwargs),
        MagicMock(url="", event_reporting=False),
    )


def _names(modules) -> list[str]:
    return [type(module).__name__ for module in modules]


def test_discovery_finds_every_builtin_module() -> None:
    found = {cls.module_name: cls for cls in discover_modules()}

    assert set(found) == EXPECTED_MODULE_NAMES
    assert all(
        dependency in found.values()
        for cls in found.values()
        for dependency in cls.module_dependencies
    )
    assert found["blend"].module_dependencies == (ld_mod.LMCacheDrivenTransferModule,)
    assert found["qstore"].module_dependencies == (ld_mod.LMCacheDrivenTransferModule,)


def test_dependency_order_survives_module_rename(monkeypatch) -> None:
    dependency = ld_mod.LMCacheDrivenTransferModule
    dependent = blend_mod.BlendModule
    monkeypatch.setattr(dependency, "module_name", "renamed_transfer")

    assert order_modules([dependent, dependency]) == [dependency, dependent]


def test_transfer_mode_gates_modules(stub_constructors) -> None:
    common = {"LookupModule", "P2PController", "ManagementModule"}
    expected = {
        "lmcache_driven": common | {"LMCacheDrivenTransferModule"},
        "engine_driven": common | {"EngineDrivenTransferModule"},
        "auto": common | {"LMCacheDrivenTransferModule", "EngineDrivenTransferModule"},
    }
    for mode, names in expected.items():
        assert set(_names(_build(supported_transfer_mode=mode))) == names


def test_management_closes_before_liveness_targets(stub_constructors) -> None:
    names = _names(_build(supported_transfer_mode="auto"))
    assert names.index("ManagementModule") < names.index("LMCacheDrivenTransferModule")


def test_blend_gating(stub_constructors) -> None:
    assert "BlendModule" not in _names(_build())
    assert "BlendModule" in _names(_build(engine_type="blend"))
    with pytest.raises(ValueError, match="blend engine requires"):
        _build(engine_type="blend", supported_transfer_mode="engine_driven")


def test_qstore_gating(stub_constructors) -> None:
    assert "QStoreModule" not in _names(_build())
    assert "QStoreModule" in _names(_build(enable=["transfer_query"]))
    with pytest.raises(ValueError, match="lmcache_driven"):
        _build(enable=["transfer_query"], supported_transfer_mode="engine_driven")
    with pytest.raises(ValueError, match="Unknown --enable"):
        _build(enable=["no_such_feature"])


def test_require_rejects_unbuilt_dependency() -> None:
    build_ctx = ModuleBuildContext(
        MagicMock(name="ctx"),
        MPServerConfig(),
        MagicMock(url="", event_reporting=False),
    )

    with pytest.raises(ValueError, match="is not built yet"):
        build_ctx.require(qstore_mod.QStoreModule)
