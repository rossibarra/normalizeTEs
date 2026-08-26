import numpy as np
import pytest

from normalize_tes.snp_position_resolution import (
    PositionResolutionError,
    resolve_native_position_requests,
    resolve_requested_positions,
)


class FakeStore:
    positions = np.array([10.0, 20.0, 30.0, 50.0])
    eligible = np.array([True, False, True, True])

    def native_to_global(self, chromosomes, positions):
        offsets = {"chr1": 0, "chr2": 40}
        return np.asarray(
            [offsets[str(chrom)] + int(pos) for chrom, pos in zip(chromosomes, positions)],
            dtype=np.float64,
        )


def test_drop_distinguishes_categories_and_preserves_request_alignment():
    resolution = resolve_requested_positions(
        FakeStore(),
        np.array([50, 15, 20, 10]),
        chromosomes=np.array(["chr2", "chr1", "chr1", "chr1"]),
        native_positions=np.array([10, 15, 20, 10]),
        policy="drop",
        label="TE positions",
    )

    np.testing.assert_array_equal(resolution.row_indices, [3, -1, 1, 0])
    np.testing.assert_array_equal(resolution.resolved_mask, [True, False, True, True])
    np.testing.assert_array_equal(resolution.eligible_mask, [True, False, False, True])
    np.testing.assert_array_equal(resolution.included_request_indices, [0, 3])
    np.testing.assert_array_equal(resolution.included_rows, [3, 0])
    np.testing.assert_array_equal(resolution.included_global_positions, [50, 10])
    np.testing.assert_array_equal(resolution.included_chromosomes, ["chr2", "chr1"])
    np.testing.assert_array_equal(resolution.included_native_positions, [10, 10])
    np.testing.assert_array_equal(resolution.unresolved_request_indices, [1])
    np.testing.assert_array_equal(resolution.unresolved_global_positions, [15])
    np.testing.assert_array_equal(resolution.ineligible_request_indices, [2])
    np.testing.assert_array_equal(resolution.ineligible_global_positions, [20])
    np.testing.assert_array_equal(resolution.excluded_request_indices, [1, 2])
    np.testing.assert_array_equal(resolution.excluded_global_positions, [15, 20])

    assert resolution.summary() == {
        "label": "TE positions",
        "policy": "drop",
        "requested_count": 4,
        "resolved_count": 3,
        "unresolved_count": 1,
        "eligible_count": 2,
        "ineligible_count": 1,
        "excluded_count": 2,
    }
    assert resolution.excluded_coordinates() == [
        {
            "request_index": 1,
            "global_position": 15,
            "reason": "unresolved",
            "chromosome": "chr1",
            "native_position": 15,
        },
        {
            "request_index": 2,
            "global_position": 20,
            "reason": "ineligible",
            "chromosome": "chr1",
            "native_position": 20,
        },
    ]


def test_error_policy_exposes_complete_audit_result():
    with pytest.raises(PositionResolutionError, match=(
        r"excluded 2 of 3 requests; unresolved=1; resolved-but-ineligible=1"
    )) as caught:
        resolve_requested_positions(
            FakeStore(), np.array([15, 30, 20]), policy="error", label="candidates"
        )

    result = caught.value.resolution
    np.testing.assert_array_equal(result.row_indices, [-1, 2, 1])
    np.testing.assert_array_equal(result.included_rows, [2])
    np.testing.assert_array_equal(result.unresolved_global_positions, [15])
    np.testing.assert_array_equal(result.ineligible_global_positions, [20])


def test_native_resolution_converts_then_keeps_native_coordinates():
    result = resolve_native_position_requests(
        FakeStore(),
        np.array(["chr2", "chr1", "chr1"]),
        np.array([10, 30, 15]),
        policy="drop",
        label="synonymous candidates",
    )
    np.testing.assert_array_equal(result.global_positions, [50, 30, 15])
    np.testing.assert_array_equal(result.included_rows, [3, 2])
    np.testing.assert_array_equal(result.included_chromosomes, ["chr2", "chr1"])
    np.testing.assert_array_equal(result.included_native_positions, [10, 30])
    assert result.excluded_coordinates() == [{
        "request_index": 2,
        "global_position": 15,
        "reason": "unresolved",
        "chromosome": "chr1",
        "native_position": 15,
    }]


@pytest.mark.parametrize("policy", ["error", "drop"])
def test_fails_when_no_eligible_positions_remain(policy):
    with pytest.raises(PositionResolutionError) as caught:
        resolve_requested_positions(FakeStore(), np.array([15, 20]), policy=policy)
    assert caught.value.resolution.eligible_count == 0
    assert caught.value.resolution.unresolved_count == 1
    assert caught.value.resolution.ineligible_count == 1


def test_rejects_empty_duplicate_and_misaligned_requests():
    with pytest.raises(ValueError, match="empty"):
        resolve_requested_positions(FakeStore(), np.array([]), policy="drop")
    with pytest.raises(ValueError, match="duplicate global positions"):
        resolve_requested_positions(FakeStore(), np.array([10, 10]), policy="drop")
    with pytest.raises(ValueError, match="aligned"):
        resolve_requested_positions(
            FakeStore(),
            np.array([10, 30]),
            chromosomes=np.array(["chr1"]),
            native_positions=np.array([10]),
            policy="drop",
        )


def test_rejects_invalid_store_and_policy_inputs():
    with pytest.raises(ValueError, match="policy"):
        resolve_requested_positions(FakeStore(), np.array([10]), policy="ignore")

    class BadPositions(FakeStore):
        positions = np.array([10.0, 10.0])
        eligible = np.ones(2, dtype=bool)

    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_requested_positions(BadPositions(), np.array([10]), policy="drop")

    class BadEligibility(FakeStore):
        eligible = np.ones(4, dtype=np.uint8)

    with pytest.raises(ValueError, match="eligible mask"):
        resolve_requested_positions(BadEligibility(), np.array([10]), policy="drop")
