from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import Tensor, nn

from cogni_core.backbone import (
    DecoderLayerContractError,
    GemmaDEQBackboneAdapter,
    LocalGemmaFeatureBackbone,
    OfflineModelPolicyError,
    inject_gemma_deq_layer,
    load_local_gemma,
    verify_local_gemma_path,
)
from cogni_core.resources import MAX_VRAM_GIB
from cogni_core.deq import ContractivityError, DEQConfig


class _HFLikeTupleLayer(nn.Module):
    def __init__(self, width: int, scale: float = 0.22) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(width) * scale)
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        position_embeddings: object | None = None,
        use_cache: bool = True,
        output_attentions: bool = True,
        past_key_value: object = "unset",
    ) -> tuple[Tensor]:
        self.calls.append(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "position_embeddings": position_embeddings,
                "use_cache": use_cache,
                "output_attentions": output_attentions,
                "past_key_value": past_key_value,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return (torch.tanh(self.proj(hidden_states)),)


class _BareTensorLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width) * 0.15)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return torch.tanh(hidden_states @ self.weight.T)


class _BadShapeLayer(nn.Module):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states[..., :-1]


class _FeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"use_cache": True})()
        self.last_kwargs = None

    def forward(self, hidden_states: Tensor, **kwargs):
        self.last_kwargs = kwargs
        return type(
            "Output", (), {"hidden_states": (hidden_states, hidden_states + 1)}
        )()


class _ResidualLayer(nn.Module):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + 0.2 * torch.tanh(hidden_states)


class _NoFixedPointLayer(nn.Module):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + 1.0


class TestGemmaDEQBackboneAdapter(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)

    def test_hf_tuple_kwargs_no_grad_solve_and_ift_match_unroll(self) -> None:
        width = 4
        cfg = DEQConfig(
            tolerance=1e-9,
            max_iter=80,
            history=6,
            fallback_steps=100,
        )
        implicit_layer = _HFLikeTupleLayer(width).double()
        explicit_layer = _HFLikeTupleLayer(width).double()
        explicit_layer.load_state_dict(implicit_layer.state_dict())
        adapter = GemmaDEQBackboneAdapter(implicit_layer, cfg).double()

        x_implicit = torch.randn(2, 3, width, dtype=torch.double, requires_grad=True)
        x_explicit = x_implicit.detach().clone().requires_grad_(True)
        mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
        positions = torch.arange(3).expand(2, -1)
        rope = (torch.randn(2, 3, width), torch.randn(2, 3, width))

        output = adapter(
            x_implicit,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=rope,
            use_cache=True,
            output_attentions=True,
            past_key_value=object(),
        )
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 1)
        self.assertTrue(adapter.last_info is not None and adapter.last_info.converged)

        # Every root-solver call is detached; only the single IFT bridge call
        # at the end of forward is grad-enabled.
        grad_modes_before_backward = [
            bool(call["grad_enabled"]) for call in implicit_layer.calls
        ]
        self.assertTrue(all(not mode for mode in grad_modes_before_backward[:-1]))
        self.assertTrue(grad_modes_before_backward[-1])
        self.assertLessEqual(
            len(grad_modes_before_backward), cfg.max_iter + cfg.fallback_steps + 3
        )
        for call in implicit_layer.calls:
            self.assertIs(call["attention_mask"], mask)
            self.assertIs(call["position_ids"], positions)
            self.assertIs(call["position_embeddings"], rope)
            self.assertFalse(call["use_cache"])
            self.assertFalse(call["output_attentions"])
            self.assertIsNone(call["past_key_value"])

        implicit_loss = output[0].square().sum()
        implicit_loss.backward()

        z = torch.zeros_like(x_explicit)
        for _ in range(300):
            z = torch.tanh(explicit_layer.proj(z + x_explicit))
        z.square().sum().backward()

        self.assertTrue(torch.allclose(output[0], z, atol=2e-7, rtol=2e-7))
        self.assertTrue(
            torch.allclose(x_implicit.grad, x_explicit.grad, atol=2e-6, rtol=2e-5)
        )
        self.assertTrue(
            torch.allclose(
                implicit_layer.proj.weight.grad,
                explicit_layer.proj.weight.grad,
                atol=3e-6,
                rtol=3e-5,
            )
        )
        self.assertTrue(adapter.last_backward_info is not None)
        self.assertTrue(adapter.last_backward_info.converged)
        self.assertLessEqual(adapter.last_backward_info.iterations, cfg.max_iter)

    def test_bare_tensor_layer_filters_unsupported_hf_kwargs(self) -> None:
        layer = _BareTensorLayer(5)
        adapter = GemmaDEQBackboneAdapter(
            layer,
            DEQConfig(tolerance=1e-7, max_iter=50, history=4),
        )
        output = adapter(
            torch.randn(2, 5),
            attention_mask=torch.ones(2, 2),
            position_ids=torch.arange(2),
            use_cache=True,
        )
        self.assertIsInstance(output, Tensor)
        self.assertEqual(tuple(output.shape), (2, 5))
        self.assertTrue(torch.isfinite(output).all())

    def test_model_output_hidden_state_is_supported(self) -> None:
        class Output:
            hidden_states = (torch.zeros(1, 2, 3), torch.ones(1, 2, 3))

        from cogni_core.backbone import extract_hidden_states

        self.assertTrue(
            torch.equal(extract_hidden_states(Output()), Output.hidden_states[-1])
        )

    def test_local_feature_backbone_forces_hidden_states_and_no_cache(self) -> None:
        model = _FeatureModel()
        wrapper = LocalGemmaFeatureBackbone(model)
        result = wrapper(torch.zeros(1, 2, 3), use_cache=True)
        self.assertTrue(torch.equal(result, torch.ones(1, 2, 3)))
        self.assertFalse(model.last_kwargs["use_cache"])
        self.assertTrue(model.last_kwargs["output_hidden_states"])

    def test_shape_change_is_rejected(self) -> None:
        adapter = GemmaDEQBackboneAdapter(_BadShapeLayer(), DEQConfig(max_iter=2))
        with self.assertRaises(DecoderLayerContractError):
            adapter(torch.randn(1, 3, 4))

    def test_contractive_delta_mode_removes_identity_residual(self) -> None:
        adapter = GemmaDEQBackboneAdapter(
            _ResidualLayer(),
            DEQConfig(tolerance=1e-6, max_iter=40),
            contractive_delta_scale=0.1,
            # delta(h) = 0.2*tanh(h), hence Lip(delta) <= 0.2.
            certified_delta_lipschitz_bound=0.2,
        )
        x = torch.randn(2, 4)
        output = adapter(x)
        self.assertTrue(adapter.last_info.converged)
        self.assertTrue(
            torch.allclose(output, x + 0.02 * torch.tanh(output + x), atol=2e-5)
        )
        self.assertAlmostEqual(adapter.last_info.spectral_norm, 0.02, places=7)

    def test_contractive_delta_requires_a_certificate_in_fail_closed_mode(self):
        with self.assertRaises(ContractivityError):
            GemmaDEQBackboneAdapter(
                _ResidualLayer(),
                contractive_delta_scale=0.1,
            )

    def test_unsafe_certified_delta_bound_is_rejected(self):
        adapter = GemmaDEQBackboneAdapter(
            _ResidualLayer(),
            contractive_delta_scale=0.5,
            certified_delta_lipschitz_bound=2.0,
        )
        with self.assertRaises(ContractivityError):
            adapter(torch.zeros(1, 4))

    def test_unconverged_injected_solve_fails_closed_by_default(self):
        class _MiniModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([_NoFixedPointLayer()])

        model = _MiniModel()
        adapter = inject_gemma_deq_layer(
            model,
            config=DEQConfig(max_iter=2, fallback_steps=2, tolerance=1e-8),
        )
        self.assertTrue(adapter.config.fail_on_noncontractive)
        with self.assertRaises(ContractivityError):
            model.model.layers[0](torch.zeros(1, 3))

    def test_injection_replaces_only_selected_decoder_layer(self) -> None:
        class _MiniModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList(
                    [_BareTensorLayer(3), _BareTensorLayer(3)]
                )

        model = _MiniModel()
        adapter = inject_gemma_deq_layer(
            model,
            layer_index=-1,
            config=DEQConfig(max_iter=30, tolerance=1e-6),
            freeze_other_parameters=True,
        )
        self.assertIs(model.model.layers[-1], adapter)
        self.assertFalse(model.model.layers[0].weight.requires_grad)
        self.assertTrue(adapter.decoder_layer.weight.requires_grad)

    def test_injection_supports_real_gemma4_language_model_layout(self) -> None:
        class _Gemma4Layout(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.layers = nn.ModuleList(
                    [_BareTensorLayer(3), _BareTensorLayer(3)]
                )

        model = _Gemma4Layout()
        adapter = inject_gemma_deq_layer(model, layer_index=0)
        self.assertIs(model.model.language_model.layers[0], adapter)


class _RecordingModelFactory:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> object:
        cls.calls.append((path, kwargs))
        return {"kind": "model", "path": path}


class _RecordingTokenizerFactory:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> object:
        cls.calls.append((path, kwargs))
        return {"kind": "tokenizer", "path": path}


def _create_local_gemma(root: Path) -> None:
    (root / "config.json").write_text(
        json.dumps({"model_type": "gemma4"}), encoding="utf-8"
    )
    (root / "model.safetensors").write_bytes(b"offline weights placeholder")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")


class TestOfflineGemmaLoader(unittest.TestCase):
    def setUp(self) -> None:
        _RecordingModelFactory.calls.clear()
        _RecordingTokenizerFactory.calls.clear()

    def test_hub_id_and_url_are_rejected_before_loading(self) -> None:
        for source in (
            "google/gemma-4-e4b",
            "https://huggingface.co/google/gemma-4-e4b",
        ):
            with (
                self.subTest(source=source),
                self.assertRaises(OfflineModelPolicyError),
            ):
                load_local_gemma(
                    source,
                    model_class=_RecordingModelFactory,
                    tokenizer_class=_RecordingTokenizerFactory,
                )
        self.assertEqual(_RecordingModelFactory.calls, [])
        self.assertEqual(_RecordingTokenizerFactory.calls, [])

    def test_verification_requires_gemma_config_weights_and_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"model_type": "llama"}), encoding="utf-8"
            )
            with self.assertRaises(OfflineModelPolicyError):
                verify_local_gemma_path(root)

            (root / "config.json").write_text(
                json.dumps({"model_type": "gemma"}), encoding="utf-8"
            )
            with self.assertRaises(OfflineModelPolicyError):
                verify_local_gemma_path(root)

            (root / "model.safetensors").write_bytes(b"weights")
            with self.assertRaises(OfflineModelPolicyError):
                verify_local_gemma_path(root)

    def test_local_loader_forces_offline_flags_and_checks_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_local_gemma(root)
            config_digest = hashlib.sha256(
                (root / "config.json").read_bytes()
            ).hexdigest()
            model, tokenizer = load_local_gemma(
                root,
                model_class=_RecordingModelFactory,
                tokenizer_class=_RecordingTokenizerFactory,
                expected_sha256={"config.json": config_digest},
                model_kwargs={"torch_dtype": "bfloat16"},
            )

            self.assertEqual(model["kind"], "model")
            self.assertEqual(tokenizer["kind"], "tokenizer")
            for calls in (
                _RecordingModelFactory.calls,
                _RecordingTokenizerFactory.calls,
            ):
                self.assertEqual(len(calls), 1)
                path, kwargs = calls[0]
                self.assertEqual(Path(path), root.resolve())
                self.assertIs(kwargs["local_files_only"], True)
                self.assertIs(kwargs["trust_remote_code"], False)
                self.assertIs(kwargs["force_download"], False)

    def test_offline_flags_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_local_gemma(root)
            with self.assertRaises(OfflineModelPolicyError):
                load_local_gemma(
                    root,
                    model_class=_RecordingModelFactory,
                    tokenizer_class=_RecordingTokenizerFactory,
                    model_kwargs={"local_files_only": False},
                )
            with self.assertRaises(OfflineModelPolicyError):
                load_local_gemma(
                    root,
                    model_class=_RecordingModelFactory,
                    tokenizer_class=_RecordingTokenizerFactory,
                    tokenizer_kwargs={"trust_remote_code": True},
                )

    def test_loader_rejects_vram_limit_above_absolute_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_local_gemma(root)
            with self.assertRaisesRegex(ValueError, str(MAX_VRAM_GIB)):
                load_local_gemma(
                    root,
                    model_class=_RecordingModelFactory,
                    tokenizer_class=_RecordingTokenizerFactory,
                    vram_limit_gib=MAX_VRAM_GIB + 0.1,
                )


if __name__ == "__main__":
    unittest.main()
