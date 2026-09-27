"""The cell-family and orientation vocabulary: closed, named, and verified against the data.

Two disciplines are under test here, both inherited from :mod:`flybrain.pet`:

1. **The vocabulary is closed.** Every family the annotation table can report has a plain-English
   name, a description and a colour, and a value that arrives without an entry becomes
   *unlabelled* rather than being guessed at. These tests must not need the connectome, so the
   published values are pinned as a literal — and a separate, skippable test checks the literal
   against the real table when it is present.
2. **Anatomy is stated and then checked, never derived by heuristic.** Which end of the brain is
   the front is a fact about this dataset, established by landmarks. ``orientation_from`` asserts
   it; ``front_back_evidence`` recomputes it from the annotations; the tests require the two to
   agree, which is the only thing stopping the constant from silently rotting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flybrain.families import (
    ALL_FAMILIES,
    FAMILIES,
    FAMILY_BY_KEY,
    SENSE_GROUPS,
    UNLABELLED,
    UNLABELLED_ID,
    families_payload,
    family_counts,
    family_ids,
    front_back_evidence,
    group_id_buffer,
    groups_payload,
    orientation_from,
    sense_ids,
    senses_payload,
)

#: The ten ``super_class`` values published in FlyWire v783, and their true counts. Pinned rather
#: than read so these tests run with no connectome; `test_the_pinned_values_match_the_real_table`
#: is what keeps the pin honest.
PUBLISHED = {
    "optic": 77530,
    "central": 32379,
    "sensory": 16352,
    "visual_projection": 8038,
    "ascending": 1736,
    "descending": 1299,
    "sensory_ascending": 581,
    "visual_centrifugal": 524,
    "motor": 110,
    "endocrine": 76,
}


class TestFamilyVocabulary:
    def test_the_vocabulary_covers_every_published_value(self) -> None:
        assert set(FAMILY_BY_KEY) == set(PUBLISHED)

    def test_unlabelled_is_id_zero_and_is_not_one_of_the_published_values(self) -> None:
        """Zero is reserved so a missing value cannot be confused with a real family."""
        assert UNLABELLED.id == UNLABELLED_ID == 0
        assert UNLABELLED.key not in PUBLISHED
        assert len(ALL_FAMILIES) == len(FAMILIES) + 1

    def test_ids_are_unique_and_fit_in_a_byte(self) -> None:
        ids = [f.id for f in ALL_FAMILIES]
        assert len(set(ids)) == len(ids)
        assert all(0 <= i <= 255 for i in ids), "ids travel as uint8 in the id buffer"

    def test_ids_do_not_change_once_shipped(self) -> None:
        """The ids are baked into the binary buffer, so reordering them is a wire-format change.

        This test is deliberately brittle: if it fails, the fix is to update the documented
        format and the client together, not to relax the assertion.
        """
        assert [f.id for f in FAMILIES] == list(range(1, len(FAMILIES) + 1))

    def test_every_family_is_explained_in_plain_english(self) -> None:
        """These names exist because `visual_projection` is not a thing a person can picture."""
        for f in ALL_FAMILIES:
            assert f.label and len(f.label) > 3, f.key
            assert f.blurb and len(f.blurb) > 20, f.key
            assert f.label != f.key or f.id == 0, f"{f.key} was not renamed for a reader"

    def test_every_family_has_a_usable_colour(self) -> None:
        for f in ALL_FAMILIES:
            assert len(f.colour) == 7 and f.colour.startswith("#"), f.key
            int(f.colour[1:], 16)

    def test_colours_are_distinct(self) -> None:
        """Two families sharing a hue would be indistinguishable in the legend and the cloud."""
        colours = [f.colour.lower() for f in ALL_FAMILIES]
        assert len(set(colours)) == len(colours)


class TestFamilyIds:
    def test_values_map_to_their_family_id(self) -> None:
        ids = family_ids(np.array(["optic", "central", "visual_projection"], dtype=object))
        assert ids.tolist() == [FAMILY_BY_KEY["optic"].id,
                                FAMILY_BY_KEY["central"].id,
                                FAMILY_BY_KEY["visual_projection"].id]

    def test_nan_becomes_unlabelled_rather_than_raising(self) -> None:
        """A rendering path must not fail because the publisher omitted a value."""
        ids = family_ids(np.array(["optic", np.nan, None], dtype=object))
        assert ids.tolist() == [FAMILY_BY_KEY["optic"].id, UNLABELLED_ID, UNLABELLED_ID]

    def test_an_unknown_value_becomes_unlabelled(self) -> None:
        """If the publisher adds a family, the picture shows a gap rather than a wrong colour."""
        assert family_ids(np.array(["a_family_from_the_future"], dtype=object)).tolist() == [0]

    def test_the_output_is_uint8(self) -> None:
        assert family_ids(np.array(["optic"], dtype=object)).dtype == np.uint8

    def test_counts_account_for_every_neuron(self) -> None:
        ids = family_ids(np.array(["optic", "optic", "motor", "nonsense"], dtype=object))
        counts = family_counts(ids)
        assert counts[FAMILY_BY_KEY["optic"].id] == 2
        assert counts[FAMILY_BY_KEY["motor"].id] == 1
        assert counts[UNLABELLED_ID] == 1
        assert sum(counts.values()) == 4

    def test_counts_include_families_with_no_members(self) -> None:
        """The legend lists all of them, so a zero is a real answer rather than a missing key."""
        counts = family_counts(np.zeros(3, dtype=np.uint8))
        assert set(counts) == {f.id for f in ALL_FAMILIES}
        assert counts[FAMILY_BY_KEY["optic"].id] == 0


class TestSenseGroups:
    def test_every_sense_is_a_real_resolvable_role(self) -> None:
        """A sense key that ``RoleResolver`` does not know would silently select nothing."""
        from flybrain.mapping import ROLE_SPECS

        for sense in SENSE_GROUPS:
            assert sense.key in ROLE_SPECS, sense.key

    def test_ids_are_unique_and_from_one(self) -> None:
        ids = [s.id for s in SENSE_GROUPS]
        assert ids == list(range(1, len(SENSE_GROUPS) + 1))

    def test_exactly_one_pathway_is_marked_trained(self) -> None:
        """Only the temperature pathway has a readout fitted on it; the others are context."""
        trained = [s.key for s in SENSE_GROUPS if s.trained]
        assert trained == ["thermosensory"]

    def test_every_sense_is_explained_in_plain_english(self) -> None:
        for s in SENSE_GROUPS:
            assert s.label and len(s.label) > 3, s.key
            assert s.blurb and len(s.blurb) > 15, s.key

    def test_sense_ids_are_written_only_onto_role_neurons(self) -> None:
        out = sense_ids({"thermosensory": np.array([1, 3])}, 5)
        assert out.tolist() == [0, 1, 0, 1, 0]

    def test_a_missing_role_leaves_its_neurons_unmarked(self) -> None:
        assert sense_ids({}, 4).tolist() == [0, 0, 0, 0]

    def test_overlapping_roles_do_not_crash(self) -> None:
        """Roles are disjoint by construction, but a second id winning is better than an error."""
        out = sense_ids({"thermosensory": np.array([0]), "visual": np.array([0])}, 2)
        assert out[0] in {s.id for s in SENSE_GROUPS}


class TestIdBuffer:
    def test_the_buffer_is_two_bytes_per_neuron_family_first(self) -> None:
        fam = np.array([1, 2, 0], dtype=np.uint8)
        sen = np.array([0, 6, 1], dtype=np.uint8)
        raw = group_id_buffer(fam, sen)
        assert len(raw) == 6
        assert list(raw) == [1, 0, 2, 6, 0, 1]

    def test_mismatched_lengths_are_refused(self) -> None:
        """Silently zipping two different lengths would mislabel every neuron after the gap."""
        with pytest.raises(ValueError, match="differ in shape"):
            group_id_buffer(np.zeros(3, dtype=np.uint8), np.zeros(2, dtype=np.uint8))


class TestOrientation:
    def test_markers_sit_on_the_ends_of_the_cloud_not_beyond_them(self) -> None:
        """Inset, not outset: the cloud is wider in z than the default camera shows, so a label
        outside the front of the brain would be projected off the edge of the viewport."""
        pos = np.array([[-0.2, -0.1, -1.0], [0.2, 0.1, 0.7], [0.0, 0.0, 0.1]], dtype=np.float32)
        # The inset is a *fraction of each axis's own span*, so a long thin cloud insets a lot in
        # z and a little in x, and the labels stay proportionate to the shape.
        m = orientation_from(pos, inset_frac=0.1)
        assert m["front"].anchor[2] == pytest.approx(-1.0 + 0.17)   # z span 1.7
        assert m["back"].anchor[2] == pytest.approx(0.7 - 0.17)
        assert m["left"].anchor[0] == pytest.approx(-0.2 + 0.04)    # x span 0.4
        assert m["right"].anchor[0] == pytest.approx(0.2 - 0.04)

    def test_markers_stay_inside_the_cloud_bounds(self) -> None:
        """Whichever way the cloud is shaped, the labels must remain within its extent."""
        pos = np.array([[-0.2, -0.1, -1.0], [0.2, 0.1, 0.7]], dtype=np.float32)
        m = orientation_from(pos)
        for key, axis, want_inside in (("front", 2, "lo"), ("back", 2, "hi"),
                                       ("left", 0, "lo"), ("right", 0, "hi")):
            lo, hi = pos[:, axis].min(), pos[:, axis].max()
            assert lo <= m[key].anchor[axis] <= hi, key

    def test_markers_are_named_for_a_reader(self) -> None:
        pos = np.zeros((2, 3), dtype=np.float32)
        m = orientation_from(pos)
        assert "front" in m["front"].label and "eyes" in m["front"].label
        assert "back" in m["back"].label

    def test_front_is_the_negative_z_end(self) -> None:
        """The one claim the whole orientation feature rests on, asserted explicitly."""
        pos = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        m = orientation_from(pos, inset_frac=0.0)
        assert m["front"].anchor[2] < m["back"].anchor[2]

    def test_an_empty_cloud_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            orientation_from(np.zeros((0, 3), dtype=np.float32))

    def test_no_claim_is_made_about_up_or_down(self) -> None:
        """The cloud is only ~0.24 deep in y and shares that axis with the camera, so a label
        there would confuse more than it explains."""
        assert set(orientation_from(np.zeros((2, 3), dtype=np.float32))) == {
            "front", "back", "left", "right",
        }

    def test_evidence_reports_none_rather_than_zero_for_a_missing_landmark(self) -> None:
        """Missing must not read as "at z = 0", which would look like a real measurement."""
        evidence = front_back_evidence(np.array(["nothing_matches"]), np.zeros((1, 3), np.float32))
        assert evidence["retina_r1_6"] is None
        assert evidence["antennal_trn"] is None

    def test_evidence_finds_the_landmarks_that_settle_the_question(self) -> None:
        types = np.array(["R1-6", "TRN", "KCg"], dtype=object)
        pos = np.array([[0, 0, -0.2], [0, 0, -0.8], [0, 0, 0.5]], dtype=np.float32)
        evidence = front_back_evidence(types, pos)
        assert evidence["retina_r1_6"] == pytest.approx(-0.2)
        assert evidence["antennal_trn"] == pytest.approx(-0.8)


class TestPayloads:
    def _ids(self, n: int = 6):
        fam = np.array([1, 1, 2, 0, 0, 0], dtype=np.uint8)
        sen = np.array([1, 0, 6, 0, 0, 0], dtype=np.uint8)
        return fam, sen

    def test_the_family_payload_accounts_for_every_neuron(self) -> None:
        fam, _ = self._ids()
        payload = families_payload(fam)
        assert sum(f["neurons"] for f in payload) == len(fam)

    def test_the_family_payload_is_ordered_by_id_with_unlabelled_present(self) -> None:
        fam, _ = self._ids()
        payload = families_payload(fam)
        assert [f["id"] for f in payload] == [f.id for f in ALL_FAMILIES]
        assert payload[-1]["key"] == "unlabelled"

    def test_the_sense_payload_reports_wired_rather_than_filtering(self) -> None:
        """A pathway with no sensor is still real cells, so hiding it would disagree with the
        brain on screen."""
        _, sen = self._ids()
        payload = senses_payload(sen, {"thermosensory": True})
        assert len(payload) == len(SENSE_GROUPS)
        wired = {s["key"]: s["wired"] for s in payload}
        assert wired["thermosensory"] is True
        assert wired["gustatory"] is False
        assert next(s for s in payload if s["key"] == "visual")["neurons"] == 1

    def test_the_groups_payload_carries_everything_the_client_needs(self) -> None:
        fam, sen = self._ids()
        pos = np.array([[0, 0, -1], [0, 0, 0.5], [0.1, 0, 0]], dtype=np.float32)
        payload = groups_payload(
            n_neurons=len(fam), family=fam, sense=sen, positions=pos,
            cell_type=np.array(["R1-6", "x", "y"], dtype=object),
        )
        assert payload["n"] == len(fam)
        assert set(payload) == {"n", "families", "senses", "orientation", "evidence"}
        assert set(payload["orientation"]) == {"front", "back", "left", "right"}
        assert payload["orientation"]["front"]["anchor"][2] < 0
        assert payload["evidence"]["retina_r1_6"] is not None

    def test_evidence_is_omitted_when_no_cell_types_are_given(self) -> None:
        """Absent beats an empty dict that a client might read as "checked and found nothing"."""
        fam, sen = self._ids()
        payload = groups_payload(
            n_neurons=len(fam), family=fam, sense=sen,
            positions=np.zeros((3, 3), dtype=np.float32),
        )
        assert "evidence" not in payload


# ------------------------------------------------------------------ against the real table

_ANNOTATIONS = Path("data/annotations/flywire_annotations_supl1.indexed.parquet")


@pytest.fixture(scope="module")
def table():
    """The indexed annotation table, or a skip when the connectome has not been fetched."""
    if not _ANNOTATIONS.exists():
        pytest.skip("annotation table not fetched")
    pd = pytest.importorskip("pandas")
    return pd.read_parquet(_ANNOTATIONS)


@pytest.mark.skipif(not _ANNOTATIONS.exists(), reason="annotation table not fetched")
class TestAgainstTheRealTable:
    """The pin, checked against the thing it stands for.

    Skipped without the connectome data — which is the normal case for a fresh checkout and for
    CI — so nothing here is load-bearing for the fast test run.
    """

    def test_the_pinned_values_match_the_real_table(self, table) -> None:
        counts = table["super_class"].value_counts().to_dict()
        for key, n in PUBLISHED.items():
            assert int(counts.get(key, 0)) == n, key

    def test_every_published_value_has_a_name_here(self, table) -> None:
        """This is the test that catches a publisher adding an eleventh family."""
        present = set(table["super_class"].dropna().unique())
        assert present <= set(FAMILY_BY_KEY), f"unnamed: {present - set(FAMILY_BY_KEY)}"

    def test_the_real_buffer_is_two_bytes_for_every_neuron(self, table) -> None:
        fam = family_ids(table["super_class"].to_numpy())
        sentinel = np.zeros(len(table), dtype=np.uint8)
        assert len(group_id_buffer(fam, sentinel)) == len(table) * 2

    def test_unlabelled_is_a_tiny_remainder_not_a_failed_join(self, table) -> None:
        """Only a handful of neurons lack a super_class, so a large remainder here would mean the
        join onto connectome order had gone wrong rather than that the fly is unlabelled."""
        fam = family_ids(table["super_class"].to_numpy())
        unlabelled = int((fam == UNLABELLED_ID).sum())
        assert unlabelled < 50, f"{unlabelled} unlabelled neurons looks like a broken join"

    def test_the_landmarks_agree_that_front_is_negative_z(self, table) -> None:
        """The claim ``orientation_from`` states as a constant, verified from the annotations.

        The retina is the front of the head and the antennal thermoreceptors sit in front of the
        brain, so both must be on the negative side. If this ever fails, the constant is wrong and
        every marker on screen is pointing the wrong way.
        """
        pos = np.load("data/codex/positions_normalized.npy").astype(np.float32)
        evidence = front_back_evidence(table["cell_type"].to_numpy(), pos)
        assert evidence["retina_r1_6"] is not None
        assert evidence["antennal_trn"] is not None
        assert evidence["retina_r1_6"] < 0, "photoreceptors must be at the front (-z)"
        assert evidence["antennal_trn"] < 0, "antennal receptors must be in front of the brain"
        assert evidence["antennal_trn"] < evidence["retina_r1_6"], "the antennae are furthest front"

    def test_left_is_negative_x_according_to_the_published_sides(self, table) -> None:
        """The left/right markers are the annotation table's own `side` column, checked."""
        pos = np.load("data/codex/positions_normalized.npy").astype(np.float32)
        side = table["side"].fillna("").astype(str).str.lower().to_numpy()
        left, right = side == "left", side == "right"
        assert left.sum() and right.sum()
        assert pos[left, 0].mean() < 0 < pos[right, 0].mean()
