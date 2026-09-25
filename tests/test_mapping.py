"""Role resolution against a synthetic annotation table — no connectome required.

This file exists mainly because of :data:`flybrain.mapping.ROLE_SPECS` ``"clock"``. The role
was originally declared as ``cell_class == "clock"``, which resolves to **zero** neurons in the
real connectome, so every consumer of the role silently saw an empty population. The cells are
present but carry a different class, and the names that *look* like clock neurons belong to a
different system entirely.
"""

from __future__ import annotations

import pandas as pd
import pytest

from flybrain.mapping import ROLE_SPECS, RoleResolver


def _table() -> pd.DataFrame:
    """A miniature annotation table containing the real naming trap.

    Rows 0-1 are ``ALLN`` antennal-lobe local neurons whose ``cell_type`` starts with ``lLN``.
    Rows 2-4 are genuine circadian clock cells.
    """
    return pd.DataFrame(
        {
            "cell_class": ["ALLN", "ALLN", None, None, "bilateral"],
            "super_class": ["central", "central", "central", "visual_projection", "optic"],
            "cell_type": ["lLN1_bc", "lLN2P_a", "DN1pA", "s-LNv_b", "l-LNv"],
        }
    )


def test_clock_filters_on_cell_type_not_cell_class() -> None:
    """Guards the original bug: ``cell_class == "clock"`` matches nothing."""
    assert ROLE_SPECS["clock"]["column"] == "cell_type"


def test_clock_spec_excludes_antennal_lobe_local_neurons() -> None:
    """``lLN*`` looks like "large lateral neuron" and is not a clock cell.

    A prefix match on the real connectome pulls in 158 antennal-lobe local neurons
    (``cell_class == "ALLN"``, i.e. olfactory interneurons) while still missing the clock.
    """
    offenders = [v for v in ROLE_SPECS["clock"]["values"] if v.startswith("lLN")]
    assert not offenders, f"clock role must not match antennal-lobe lLN types: {offenders}"


def test_clock_role_selects_only_the_circadian_cells() -> None:
    resolver = RoleResolver(_table(), n_neurons=5)
    assert set(resolver.resolve("clock").indices.tolist()) == {2, 3, 4}


def test_clock_role_is_not_empty() -> None:
    """The whole point of the fix: the role must actually resolve to something."""
    resolver = RoleResolver(_table(), n_neurons=5)
    assert len(resolver.resolve("clock").indices) == 3


def test_unknown_role_is_rejected() -> None:
    resolver = RoleResolver(_table(), n_neurons=5)
    with pytest.raises(KeyError):
        resolver.resolve("definitely_not_a_role")


def test_every_role_spec_has_a_description() -> None:
    """Roles are documented where they are defined; an undescribed role is a future bug."""
    missing = [name for name, spec in ROLE_SPECS.items() if not spec.get("description")]
    assert not missing, f"roles without a description: {missing}"
