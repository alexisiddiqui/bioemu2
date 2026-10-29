"""Template-conditioned equilibrium sampling: per-frame lags through the propagation pathway."""

from types import SimpleNamespace
from typing import Any, cast

import mdtraj
import numpy as np
import pytest
import torch

import bioemu2.propagation as inference_api
from bioemu2 import EDMSampler, GuidedFKSampler, propagate_with_embeddings
from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.propagation import sample_from_template_with_embeddings, template_lag_times
from bioemu2.structure import BatchAllAtomStructure


def _three_atom_topology() -> mdtraj.Topology:
    topology = mdtraj.Topology()
    residue = topology.add_residue("ALA", topology.add_chain())
    for name in ("N", "CA", "CB"):
        topology.add_atom(name, mdtraj.element.carbon, residue)
    return topology


def _setup(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]):
    topology = _three_atom_topology()
    batch = BatchAllAtomStructure(
        coords_cond=torch.zeros(1, 4, 3),
        is_dynamics_conditioned=torch.zeros(1, 1, dtype=torch.bool),
        dt=torch.zeros(1, 1),
        atom_pad_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    def sample(**kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        return kwargs["batch"].coords_cond[:, :3]

    monkeypatch.setattr(inference_api, "generate_samples", sample)
    monkeypatch.setattr(
        inference_api, "prepare_sampling_inputs", lambda *_args, **_kwargs: (topology, batch)
    )
    system: Any = SimpleNamespace(topology=topology, bonds=[])
    model: Any = SimpleNamespace(device=torch.device("cpu"), supports_propagation=True)
    template = mdtraj.Trajectory(
        xyz=np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]], dtype=np.float32),
        topology=topology,
    )
    return system, cast(TrunkEmbeddings, SimpleNamespace()), model, template


def test_template_lag_times_fixed_and_log_uniform() -> None:
    np.testing.assert_array_equal(template_lag_times(3, 10.0), [10.0, 10.0, 10.0])
    lags = template_lag_times(2000, 10.0, 1e5, seed=0)
    assert lags.min() >= 10.0 and lags.max() <= 1e5
    # Log-uniform: half the draws fall below the geometric midpoint (1000 ns).
    assert abs((lags < 1000.0).mean() - 0.5) < 0.05
    np.testing.assert_array_equal(lags, template_lag_times(2000, 10.0, 1e5, seed=0))


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((0, 1.0), "num_samples"),
        ((2, 0.0), "lag_time_ns"),
        ((2, 10.0, 1.0), "max_lag_time_ns"),
        ((2, 10.0, float("inf")), "max_lag_time_ns"),
    ],
)
def test_template_lag_times_validates(args: tuple, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        template_lag_times(*args)


def test_propagate_takes_one_lag_per_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    system, trunk, model, template = _setup(monkeypatch, calls)
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


def test_sample_from_template_records_lags_and_rejects_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    system, trunk, model, template = _setup(monkeypatch, calls)
    samples = sample_from_template_with_embeddings(
        system,
        trunk,
        template=template,
        num_samples=5,
        lag_time_ns=10.0,
        max_lag_time_ns=1e4,
        model=model,
        sampler=EDMSampler(),
        seed=3,
    )
    assert samples.n_frames == 5
    np.testing.assert_allclose(samples.time, template_lag_times(5, 10.0, 1e4, seed=3) * 1000.0)
    assert all(bool(c["batch"].is_dynamics_conditioned.all()) for c in calls)
    with pytest.raises(ValueError, match="Resampling samplers mix particles across lags"):
        sample_from_template_with_embeddings(
            system,
            trunk,
            template=template,
            num_samples=4,
            lag_time_ns=10.0,
            max_lag_time_ns=1e4,
            model=model,
            sampler=GuidedFKSampler(resampling_interval=1),
        )
