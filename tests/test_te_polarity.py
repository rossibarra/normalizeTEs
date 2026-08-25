"""Focused tests for the TE-polarity path.

Code review round 9 observed that nothing exercised `masked_row_cdfs()`,
`load_polarity_selection()`, the mask builder's draw-id mapping, or the
authentication of an ancestral table against its store, so the established
suite could pass with any of them broken. These are the regression tests for
the failures that review identified.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import build_ancestral_states
import build_te_polarity_mask as maskbuilder
import te_age_target
from snp_interval_dataset import IntervalBatch


def _interval_store(intervals_by_row, *, n_draws=3, digest="store-digest",
                    inputs=None):
    """A store returning fixed intervals, enough for the masked CDF path.

    `intervals_by_row` maps row -> list of (below, above, draw_id).
    """
    if inputs is None:
        inputs = [{"draw_id": i, "path": f"/draws/run.{i}.tsz"} for i in range(n_draws)]

    def intervals(rows):
        rows = np.asarray(rows, dtype=np.int64)
        below, above, draw = [], [], []
        offsets = [0]
        for row in rows:
            entries = intervals_by_row.get(int(row), [])
            for b, a, d in entries:
                below.append(b); above.append(a); draw.append(d)
            offsets.append(len(below))
        return IntervalBatch(
            rows=rows,
            offsets=np.asarray(offsets, dtype=np.int64),
            below=np.asarray(below, dtype=np.float64),
            above=np.asarray(above, dtype=np.float64),
            draw_id=np.asarray(draw, dtype=np.int64),
        )

    return SimpleNamespace(
        intervals=intervals,
        positions=np.arange(max(intervals_by_row) + 1 if intervals_by_row else 1,
                            dtype=np.float64),
        metadata={"content_sha256": digest, "n_posterior_draws": n_draws,
                  "inputs": inputs},
    )


def _write_mask(path, agrees, present, rows, *, digest="store-digest",
                covered=None, schema="te-polarity-mask-v1", complete=True):
    path.mkdir(parents=True)
    np.save(path / "agrees_with_biology.npy", np.asarray(agrees, dtype=bool))
    np.save(path / "draw_present.npy", np.asarray(present, dtype=bool))
    np.save(path / "te_row_indices.npy", np.asarray(rows, dtype=np.int64))
    if covered is None:
        covered = list(range(np.asarray(agrees).shape[1]))
    (path / "metadata.json").write_text(json.dumps({
        "schema_version": schema,
        "complete": complete,
        "store_content_sha256": digest,
        "covered_draw_ids": covered,
        "n_draws": np.asarray(agrees).shape[1],
    }), encoding="utf-8")


# --- finding 1: an all-flipped usable set must not yield a NaN CDF ----------

def test_masked_cdf_falls_back_when_every_usable_interval_is_flipped():
    """The fallback must be decided on intervals, not on the mask.

    Row 0 has agreeing draws in the mask, but the only draw that actually
    supplied an interval (draw 2) is flipped. Testing `keep[i].any()` passes
    here and leaves an empty row, which `_batch_cdf` returns as all-NaN.
    """
    store = _interval_store({0: [(10.0, 20.0, 2)]}, n_draws=3)
    keep = np.array([[True, True, False]])          # draws 0,1 agree; 2 does not
    edges = np.array([0.0, 15.0, 30.0, 60.0])
    cdfs = te_age_target.masked_row_cdfs(store, np.array([0]), edges, keep)
    assert np.isfinite(cdfs).all(), "masked CDF must never contain NaN"
    assert cdfs.shape == (1, edges.size)


def test_masked_cdf_drops_only_the_flipped_intervals():
    store = _interval_store({0: [(10.0, 20.0, 0), (1000.0, 2000.0, 2)]}, n_draws=3)
    edges = np.array([0.0, 30.0, 60.0, 5000.0])
    keep_all = np.array([[True, True, True]])
    keep_one = np.array([[True, True, False]])
    both = te_age_target.masked_row_cdfs(store, np.array([0]), edges, keep_all)
    only_young = te_age_target.masked_row_cdfs(store, np.array([0]), edges, keep_one)
    # Dropping the old interval must move mass earlier, not leave the CDF alone.
    assert only_young[0, 1] > both[0, 1]
    assert np.isfinite(only_young).all()


def test_masked_cdf_rejects_a_row_with_no_intervals_at_all():
    store = _interval_store({0: []}, n_draws=2)
    keep = np.array([[True, True]])
    with pytest.raises(ValueError, match="non-finite age CDF"):
        te_age_target.masked_row_cdfs(
            store, np.array([0]), np.array([0.0, 1.0, 2.0]), keep)


# --- finding 5 / coverage: the mask must authenticate against the store -----

def test_selection_rejects_a_partially_covering_mask(tmp_path):
    """An uncovered draw is not a flipped draw and must not be treated as one."""
    store = _interval_store({0: [(1.0, 2.0, 0)]}, n_draws=3)
    mask = tmp_path / "mask"
    _write_mask(mask, [[True, True, False]], [[True, True, False]], [0],
                covered=[0, 1])                      # draw 2 never examined
    with pytest.raises(SystemExit, match="posterior draws"):
        te_age_target.load_polarity_selection(mask, np.array([0]), store, None)


def test_selection_rejects_a_mask_built_against_another_store(tmp_path):
    store = _interval_store({0: [(1.0, 2.0, 0)]}, n_draws=2, digest="store-a")
    mask = tmp_path / "mask"
    _write_mask(mask, [[True, True]], [[True, True]], [0], digest="store-b")
    with pytest.raises(SystemExit, match="different store"):
        te_age_target.load_polarity_selection(mask, np.array([0]), store, None)


def test_selection_rejects_permuted_rows(tmp_path):
    """Alignment is positional, so a permutation must not be silently accepted."""
    store = _interval_store({0: [(1.0, 2.0, 0)], 1: [(1.0, 2.0, 0)]}, n_draws=2)
    mask = tmp_path / "mask"
    _write_mask(mask, [[True, True], [True, True]],
                [[True, True], [True, True]], [1, 0])
    with pytest.raises(SystemExit, match="do not match"):
        te_age_target.load_polarity_selection(
            mask, np.array([0, 1]), store, None)


def test_max_flipped_fraction_discards_only_sites_above_the_threshold(tmp_path):
    store = _interval_store({0: [], 1: [], 2: []}, n_draws=4)
    mask = tmp_path / "mask"
    agrees = [[True, True, True, True],      # 0.00 flipped
              [True, True, True, False],     # 0.25 flipped
              [True, False, False, False]]   # 0.75 flipped
    present = [[True] * 4] * 3
    _write_mask(mask, agrees, present, [0, 1, 2])
    sel = te_age_target.load_polarity_selection(
        mask, np.array([0, 1, 2]), store, 0.5)
    assert sel.keep_sites.tolist() == [True, True, False]
    assert sel.report["sites_discarded_by_threshold"] == 1
    assert sel.keep_draws.shape[0] == 2


def test_selection_without_a_threshold_keeps_every_site(tmp_path):
    store = _interval_store({0: [], 1: []}, n_draws=2)
    mask = tmp_path / "mask"
    _write_mask(mask, [[True, True], [False, False]], [[True, True]] * 2, [0, 1])
    sel = te_age_target.load_polarity_selection(mask, np.array([0, 1]), store, None)
    assert sel.keep_sites.all()
    # A site with no agreeing draw keeps all of its draws rather than none.
    assert sel.keep_draws[1].all()
    assert sel.report["sites_with_no_agreeing_draw"] == 1


# --- the mask builder must index columns by the store's draw_id ------------

def test_mask_columns_follow_store_draw_ids_not_argument_order(tmp_path):
    store = _interval_store({0: []}, n_draws=3)
    files = [tmp_path / "run.2.tsz", tmp_path / "run.0.tsz"]
    inputs = [{"draw_id": 0, "path": str(tmp_path / "run.0.tsz")},
              {"draw_id": 1, "path": str(tmp_path / "run.1.tsz")},
              {"draw_id": 2, "path": str(tmp_path / "run.2.tsz")}]
    store.metadata["inputs"] = inputs
    n_draws, columns = maskbuilder.store_draw_columns(store, files)
    assert n_draws == 3
    assert columns == [2, 0], "columns must come from the store, not argv order"


def test_mask_builder_rejects_a_tree_the_store_does_not_know(tmp_path):
    store = _interval_store({0: []}, n_draws=1)
    store.metadata["inputs"] = [{"draw_id": 0, "path": str(tmp_path / "known.tsz")}]
    with pytest.raises(SystemExit, match="not one of the store's"):
        maskbuilder.store_draw_columns(store, [tmp_path / "stranger.tsz"])


def test_mask_builder_rejects_a_repeated_tree(tmp_path):
    store = _interval_store({0: []}, n_draws=1)
    path = tmp_path / "known.tsz"
    store.metadata["inputs"] = [{"draw_id": 0, "path": str(path)}]
    with pytest.raises(SystemExit, match="more than once"):
        maskbuilder.store_draw_columns(store, [path, path])


# --- finding 3: the ancestral table must prove it used the store's draws ---

def test_ancestral_accumulate_rejects_foreign_draws(tmp_path):
    store = _interval_store({0: []}, n_draws=2)
    store.metadata["inputs"] = [
        {"draw_id": 0, "path": str(tmp_path / "a.tsz")},
        {"draw_id": 1, "path": str(tmp_path / "b.tsz")},
    ]
    with pytest.raises(SystemExit, match="not among the store's"):
        build_ancestral_states.accumulate(
            store, [tmp_path / "elsewhere.tsz"],
            chromosome=None, offsets={}, sequence_length=0.0,
            counts=np.zeros((1, 4), dtype=np.uint16),
            present=np.zeros(1, dtype=np.uint16), progress=False,
        )


def test_store_input_paths_requires_recorded_inputs():
    store = SimpleNamespace(metadata={"content_sha256": "x"})
    with pytest.raises(SystemExit, match="records no 'inputs'"):
        build_ancestral_states.store_input_paths(store)


# --- finding 4: candidate rows must be authenticated against their report ---

def _candidate_artifact(tmp_path, rows, *, content="store-digest",
                        catalog="catalog-digest", digest=None, count=None):
    import bootstrap_target_matcher as matcher
    path = tmp_path / "candidates.npy"
    array = np.asarray(rows, dtype=np.int64)
    np.save(path, array)
    report = {
        "store_content_sha256": content,
        "store_catalog_sha256": catalog,
        "candidate_rows": int(array.size) if count is None else count,
        "candidate_rows_sha256": (
            matcher._candidate_array_digest(array) if digest is None else digest),
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(report),
                                                       encoding="utf-8")
    return path, array


def _catalog_store(digest="store-digest", catalog="catalog-digest"):
    return SimpleNamespace(
        positions=np.arange(100, dtype=np.float64),
        metadata={"content_sha256": digest, "catalog_sha256": catalog},
    )


def test_candidate_rows_require_a_provenance_report(tmp_path):
    import bootstrap_target_matcher as matcher
    path = tmp_path / "candidates.npy"
    np.save(path, np.arange(5, dtype=np.int64))
    with pytest.raises(SystemExit, match="no provenance report"):
        matcher._authenticate_candidate_rows(
            path, _catalog_store(), np.arange(5, dtype=np.int64))


def test_candidate_rows_from_another_store_are_rejected(tmp_path):
    import bootstrap_target_matcher as matcher
    path, array = _candidate_artifact(tmp_path, [1, 2, 3], content="other-store")
    with pytest.raises(SystemExit, match="different store"):
        matcher._authenticate_candidate_rows(path, _catalog_store(), array)


def test_candidate_rows_modified_after_publication_are_rejected(tmp_path):
    import bootstrap_target_matcher as matcher
    path, _ = _candidate_artifact(tmp_path, [1, 2, 3])
    tampered = np.array([1, 2, 4], dtype=np.int64)
    np.save(path, tampered)
    with pytest.raises(SystemExit, match="does not match the digest"):
        matcher._authenticate_candidate_rows(path, _catalog_store(), tampered)


def test_candidate_rows_accept_a_matching_report(tmp_path):
    import bootstrap_target_matcher as matcher
    path, array = _candidate_artifact(tmp_path, [1, 2, 3])
    report = matcher._authenticate_candidate_rows(path, _catalog_store(), array)
    assert report["candidate_rows"] == 3


# --- finding 5: a mask that cannot prove its store must not be applied ------

def test_selection_rejects_a_mask_with_no_store_digest(tmp_path):
    store = _interval_store({0: [(1.0, 2.0, 0)]}, n_draws=2)
    mask = tmp_path / "mask"
    _write_mask(mask, [[True, True]], [[True, True]], [0], digest=None)
    with pytest.raises(SystemExit, match="missing a content digest"):
        te_age_target.load_polarity_selection(mask, np.array([0]), store, None)


def test_selection_rejects_an_internally_inconsistent_mask(tmp_path):
    store = _interval_store({0: [(1.0, 2.0, 0)]}, n_draws=2)
    mask = tmp_path / "mask"
    # agrees where the draw is recorded absent
    _write_mask(mask, [[True, True]], [[True, False]], [0])
    with pytest.raises(SystemExit, match="internally inconsistent"):
        te_age_target.load_polarity_selection(mask, np.array([0]), store, None)


def test_mask_builder_rejects_a_target_from_another_store(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    np.save(target / "te_row_indices.npy", np.array([0, 1], dtype=np.int64))
    (target / "metadata.json").write_text(json.dumps({
        "source_store_content_sha256": "other-store",
        "source_catalog_sha256": "catalog-digest",
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="different store"):
        maskbuilder.load_target_rows(target, _catalog_store())


def test_mask_builder_rejects_duplicate_target_rows(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    np.save(target / "te_row_indices.npy", np.array([3, 3], dtype=np.int64))
    (target / "metadata.json").write_text(json.dumps({
        "source_store_content_sha256": "store-digest",
        "source_catalog_sha256": "catalog-digest",
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate rows"):
        maskbuilder.load_target_rows(target, _catalog_store())


def test_mask_builder_rejects_out_of_range_target_rows(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    np.save(target / "te_row_indices.npy", np.array([0, 10_000], dtype=np.int64))
    (target / "metadata.json").write_text(json.dumps({
        "source_store_content_sha256": "store-digest",
        "source_catalog_sha256": "catalog-digest",
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="outside the store"):
        maskbuilder.load_target_rows(target, _catalog_store())
