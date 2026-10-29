"""Equilibrium sampling around a template by partial diffusion, and per-frame propagation lags."""

from types import SimpleNamespace
from typing import Any, cast

import mdtraj
import numpy as np
import pytest
import torch

import bioemu2.propagation as propagation_api
import bioemu2.sampling as sampling_api
from bioemu2 import EDMSampler, GuidedFKSampler, propagate_with_embeddings
from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.denoising.edm import edm_sample_schedule, initial_coords, truncate_schedule
from bioemu2.sampling import sample_from_template_with_embeddings
from bioemu2.structure import BatchAllAtomStructure


def _three_atom_topology() -> mdtraj.Topology:
    topology = mdtraj.Topology()
    residue = topology.add_residue("ALA", topology.add_chain())
    for name in ("N", "CA", "CB"):
        topology.add_atom(name, mdtraj.element.carbon, residue)
    return topology


def _fakes(monkeypatch: pytest.MonkeyPatch, module: Any, calls: list[dict[str, Any]], sample):
    topology = _three_atom_topology()
    batch = BatchAllAtomStructure(
        coords=torch.zeros(1, 4, 3),
        coords_cond=torch.zeros(1, 4, 3),
        is_dynamics_conditioned=torch.zeros(1, 1, dtype=torch.bool),
        dt=torch.zeros(1, 1),
        atom_pad_mask=torch.ones(1, 4, dtype=torch.bool),
        num_tokens=torch.tensor([1]),
    )

    def fake(**kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        return sample(kwargs)

    monkeypatch.setattr(module, "generate_samples", fake)
    monkeypatch.setattr(module, "prepare_sampling_inputs", lambda *_a, **_k: (topology, batch))
    system: Any = SimpleNamespace(topology=topology, bonds=[])
    model: Any = SimpleNamespace(device=torch.device("cpu"), supports_propagation=True)
    template = mdtraj.Trajectory(
        xyz=np.array([[[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.0, 0.2, 0.0]]], dtype=np.float32),
        topology=topology,
    )
    return system, cast(TrunkEmbeddings, SimpleNamespace()), model, template


def test_truncate_schedule_keeps_levels_at_or_below_start() -> None:
    sigmas = edm_sample_schedule(200, device=torch.device("cpu"))
    assert torch.equal(truncate_schedule(sigmas, None), sigmas)
    kept = truncate_schedule(sigmas, 20.0)
    assert kept[0] <= 20.0 and kept[-1] == 0.0 and len(kept) < len(sigmas)
    torch.testing.assert_close(kept, sigmas[sigmas <= 20.0])
    with pytest.raises(ValueError, match="start_sigma"):
        truncate_schedule(sigmas, 1e-9)
    with pytest.raises(ValueError, match="start_sigma"):
        truncate_schedule(sigmas, -1.0)


def test_initial_coords_start_from_template_plus_noise() -> None:
    template = torch.arange(6.0).reshape(1, 2, 3)
    torch.manual_seed(0)
    noise = initial_coords(template, torch.tensor(2.0), None, torch.randn)
    torch.manual_seed(0)
    started = initial_coords(template, torch.tensor(2.0), 2.0, torch.randn)
    torch.testing.assert_close(started, template + noise)


def test_samplers_validate_start_sigma() -> None:
    assert EDMSampler(start_sigma=10.0).start_sigma == 10.0
    with pytest.raises(ValueError, match="start_sigma"):
        EDMSampler(start_sigma=0.0)
    with pytest.raises(ValueError, match="start_sigma"):
        GuidedFKSampler(start_sigma=float("nan"))


def test_sample_from_template_noises_the_template(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    rotation = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system, trunk, model, template = _fakes(
        monkeypatch,
        sampling_api,
        calls,
        lambda kw: kw["batch"].coords[:, :3].repeat(kw["num_samples"], 1, 1) @ rotation,
    )
    samples = sample_from_template_with_embeddings(
        system,
        trunk,
        template=template,
        start_sigma=15.0,
        model=model,
        num_samples=3,
        sampler=EDMSampler(),
        seed=1,
    )
    (call,) = calls
    assert call["sampler"].start_sigma == 15.0 and isinstance(call["sampler"], EDMSampler)
    # The template, centred and padded, is the starting point; no dynamics conditioning.
    centred = 10 * (template.xyz[0] - template.xyz[0].mean(axis=0))
    torch.testing.assert_close(call["batch"].coords[0, :3], torch.as_tensor(centred))
    assert call["batch"].coords[0, 3].abs().sum() == 0
    assert not bool(call["batch"].is_dynamics_conditioned.any())
    # Samples are superposed back on the template.
    assert samples.n_frames == 3
    np.testing.assert_allclose(samples.xyz, np.repeat(template.xyz, 3, axis=0), atol=1e-5)
    with pytest.raises(ValueError, match="EDMSampler and GuidedFKSampler"):
        sample_from_template_with_embeddings(
            system,
            trunk,
            template=template,
            start_sigma=15.0,
            model=model,
            sampler=cast(Any, SimpleNamespace()),
        )


def test_propagate_takes_one_lag_per_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    system, trunk, model, template = _fakes(
        monkeypatch, propagation_api, calls, lambda kw: kw["batch"].coords_cond[:, :3]
    )
    propagate_with_embeddings(
        system,
        trunk,
        x0=template[[0, 0, 0]],
        lag_time_ns=[1.0, 10.0, 100.0],
        model=model,
        batch_size=2,
        sampler=EDMSampler(),
    )
    dts = torch.cat([c["batch"].dt for c in calls], dim=1)
    torch.testing.assert_close(dts, torch.tensor([[1e3, 1e4, 1e5]]))
    with pytest.raises(ValueError, match="one lag per x0 frame"):
        propagate_with_embeddings(
            system, trunk, x0=template[[0, 0]], lag_time_ns=[1.0], model=model, sampler=EDMSampler()
        )
