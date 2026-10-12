# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamically loaded multiprocess server modules."""

# Future
from __future__ import annotations

# Standard
from types import ModuleType
from typing import Any, ClassVar
from unittest.mock import MagicMock
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleSpec,
    build_server_module_router,
    load_server_module_classes,
    parse_server_module_specs,
    register_grpc_services,
    register_zmq_services,
    server_module_handler,
)
from lmcache.v1.multiprocess.modules import engine_driven_transfer as ed_mod
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as ld_mod
from lmcache.v1.multiprocess.modules import lookup as lookup_mod
from lmcache.v1.multiprocess.modules import management as management_mod
from lmcache.v1.multiprocess.modules import p2p_controller as p2p_mod
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
from lmcache.v1.multiprocess.request_handler import iter_request_handlers

# Modules whose constructors are stubbed in the build tests below. Each entry
# is (defining module, class name); the classes are patched in place so the
# discovery contract stays intact.
_BUILTIN_MODULES = [
    lookup_mod.LookupModule,
    p2p_mod.P2PController,
    ld_mod.LMCacheDrivenTransferModule,
    ed_mod.EngineDrivenTransferModule,
    management_mod.ManagementModule,
]


class _FakePluginModule(DiscoverableModule):
    module_name = "fake_plugin"
    module_dependencies = (lookup_mod.LookupModule,)
    last_created: ClassVar[_FakePluginModule | None] = None
    last_dependency: ClassVar[EngineModule | None] = None

    def __init__(self, ctx, config: dict[str, Any] | None = None) -> None:
        self._ctx = ctx
        self.config = config or {}
        self.touched: list[int] = []

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        cls.last_dependency = build_ctx.require(lookup_mod.LookupModule)
        module = cls(
            build_ctx.engine_context,
            dict(build_ctx.config_for(cls)),
        )
        cls.last_created = module
        return module

    @property
    def context(self):
        return self._ctx

    def report_status(self) -> dict:
        return {"fake_plugin": {"is_healthy": True}}

    def close(self) -> None:
        return None

    def touch_instance(self, instance_id: int) -> None:
        self.touched.append(instance_id)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        return []

    def tracked_instance_count(self) -> int:
        return len(self.touched)

    def drop_instance_state(self, instance_id: int) -> None:
        return None


class _FakePluginProtocolModule(_FakePluginModule):
    module_name = "fake_plugin_protocol"

    @server_module_handler("fake.echo")
    def echo(self, payload: bytes) -> bytes:
        return b"echo:" + payload

    @server_module_handler("fake.fail")
    def fail(self, payload: bytes) -> bytes:
        raise RuntimeError(payload.decode())


class _FakePluginTransportServiceModule(_FakePluginModule):
    module_name = "fake_plugin_transport"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self.grpc_server: object | None = None
        self.zmq_server: object | None = None

    def register_grpc_services(self, server: object) -> None:
        self.grpc_server = server

    def register_zmq_services(self, server: object) -> None:
        self.zmq_server = server


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    module_type: type[DiscoverableModule],
    *,
    module_name: str = "fake_server_module_plugin",
) -> str:
    module = ModuleType(module_type.__module__)
    setattr(module, module_type.__name__, module_type)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def _stub_builtin_modules(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    def _capture(self, *args, **kwargs):
        captured[type(self).__name__] = kwargs

    for module_type in _BUILTIN_MODULES:
        monkeypatch.setattr(module_type, "__init__", _capture)
    return captured


def test_parse_server_module_specs_accepts_object_and_list() -> None:
    specs = parse_server_module_specs(
        [
            '{"module_path":"plugin.one","config":{"a":1}}',
            ('[{"module_path":"plugin.two"},{"module_path":"plugin.three"}]'),
        ]
    )

    assert [spec.module_path for spec in specs] == [
        "plugin.one",
        "plugin.two",
        "plugin.three",
    ]
    assert specs[0].config == {"a": 1}
    assert specs[1].config == {}
    assert specs[2].config == {}


def test_parse_server_module_specs_rejects_removed_factory_name() -> None:
    with pytest.raises(ValueError, match="unknown server module fields"):
        parse_server_module_specs(
            ['{"module_path":"plugin.one","factory_name":"build_server_modules"}']
        )


def test_load_server_module_classes_returns_class_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _install_fake_module(monkeypatch, _FakePluginModule)
    configured = load_server_module_classes(
        [ExtServerModuleSpec(module_name, config={"mode": "test"})]
    )

    assert configured == [(_FakePluginModule, {"mode": "test"})]


def test_load_server_module_classes_requires_discoverable_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "empty_server_module_plugin"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))

    with pytest.raises(TypeError, match="defines no concrete"):
        load_server_module_classes([ExtServerModuleSpec(module_name)])


def test_load_server_module_classes_ignores_imported_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "server_module_with_imported_dependency"
    module: Any = ModuleType(module_name)
    external_type = type(
        "ExternalModule",
        (_FakePluginModule,),
        {
            "__module__": module_name,
            "module_name": "external",
            "module_dependencies": (lookup_mod.LookupModule,),
        },
    )
    module.ExternalModule = external_type
    module.LookupModule = lookup_mod.LookupModule
    monkeypatch.setitem(sys.modules, module_name, module)

    configured = load_server_module_classes([ExtServerModuleSpec(module_name)])

    assert configured == [(external_type, {})]


def test_server_module_router_dispatches_extension_handlers() -> None:
    plugin_module = _FakePluginProtocolModule(MagicMock(name="ctx"))
    router = build_server_module_router(plugin_module.context, [plugin_module])
    assert router is not None

    registered = iter_request_handlers(router)
    assert len(registered) == 1
    assert registered[0].operation == "server_module_call"

    ok = router.server_module_call(
        ServerModuleCallRequest(method="fake.echo", payload=b"hello")
    )
    missing = router.server_module_call(
        ServerModuleCallRequest(method="fake.missing", payload=b"hello")
    )
    failed = router.server_module_call(
        ServerModuleCallRequest(method="fake.fail", payload=b"boom")
    )

    assert ok.success
    assert ok.payload == b"echo:hello"
    assert ok.error == ""
    assert not missing.success
    assert missing.payload == b""
    assert "fake.missing" in missing.error
    assert not failed.success
    assert failed.error == "boom"


def test_server_module_router_rejects_duplicate_extension_methods() -> None:
    ctx = MagicMock(name="ctx")

    with pytest.raises(ValueError, match="duplicate server module method"):
        build_server_module_router(
            ctx,
            [
                _FakePluginProtocolModule(ctx),
                _FakePluginProtocolModule(ctx),
            ],
        )


def test_build_server_module_router_returns_none_without_handlers() -> None:
    assert (
        build_server_module_router(
            MagicMock(name="ctx"),
            [_FakePluginModule(MagicMock(name="plugin_ctx"))],
        )
        is None
    )


def test_transport_service_registrars_are_called() -> None:
    module = _FakePluginTransportServiceModule(MagicMock(name="ctx"))
    grpc_server = object()
    zmq_server = object()

    register_grpc_services([module], grpc_server)
    register_zmq_services([module], zmq_server)

    assert module.grpc_server is grpc_server
    assert module.zmq_server is zmq_server


def test_transport_service_registrars_must_be_callable() -> None:
    class BadModule:
        register_grpc_services = "not-callable"

    with pytest.raises(TypeError, match="register_grpc_services must be callable"):
        register_grpc_services([BadModule()], object())


def test_build_modules_loads_plugin_and_registers_liveness_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = MagicMock(name="ctx")
    logger_info = MagicMock()
    monkeypatch.setattr(server_mod.logger, "info", logger_info)
    module_name = _install_fake_module(monkeypatch, _FakePluginModule)
    management = _stub_builtin_modules(monkeypatch)

    modules = server_mod._build_modules(
        ctx,
        MPServerConfig(
            server_modules=[ExtServerModuleSpec(module_name, config={"mode": "test"})]
        ),
        MagicMock(url=""),
    )
    plugin_module = _FakePluginModule.last_created

    assert plugin_module is not None
    assert plugin_module.config == {"mode": "test"}
    assert plugin_module in modules
    assert plugin_module in management["ManagementModule"]["liveness_targets"]
    dependency = _FakePluginModule.last_dependency
    assert isinstance(dependency, lookup_mod.LookupModule)
    assert modules.index(plugin_module) < modules.index(dependency)
    logger_info.assert_called_once()
    log_format, module_count, module_details = logger_info.call_args.args
    log_message = log_format % (module_count, module_details)
    assert "Composed 5 engine modules" in log_message
    assert "fake_plugin(dependencies=[lookup])" in log_message
