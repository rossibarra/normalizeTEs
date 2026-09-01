#!/usr/bin/env python3
"""Combined posterior age distribution of the derived alleles one individual carries.

This is a per-individual analysis, not part of the TE/control matching workflow.
It answers: across every variant an individual is genotyped at, how old are the
derived alleles that individual carries?

There is no variant-class logic here. SNPs and TEs are not separated: every
biallelic VCF record with A/C/G/T alleles that resolves to a store row enters the
mixture on the same terms, and polarity for all of them comes from the ARG's
per-draw ancestral call. That differs from `phi_sfs`, which polarizes TE sites by
biology instead. Split the VCF with `--include-positions`/`--exclude-positions`
if the classes need different handling.

What one site contributes
-------------------------
Within a single posterior draw, a biallelic variant with one mutation has one age
interval -- the branch the mutation sits on -- and one derived allele, the one
that branch's descendants carry. The draw therefore either does or does not put
a derived allele in a given individual:

* a homozygote contributes that draw's age interval only in the draws where the
  allele it carries is the derived one;
* a heterozygote carries both alleles, so *whichever* allele a draw calls
  derived, the individual carries it: every usable draw contributes, and the
  age used is that draw's age.

Averaging each draw's contribution over the row's usable draws makes a site's
total mass equal to the posterior probability that the individual carries a
derived allele there. A site whose carried allele is derived in every draw
weighs twice one that is derived in half of them, which is the intended
weighting; a heterozygous site weighs 1 because some allele is derived in every
draw.

Why per-draw polarity, not the marginal proportion
--------------------------------------------------
The weight alone could be read off `build_ancestral_states`. The *ages* could
not. Selecting the draws in which a chosen allele is derived selects a subset of
the row's intervals, and that subset is not an unbiased sample of the row's ages:
in a draw where the derived allele is the common one the mutation sits on a
deeper branch than in a draw where it is the rare one. Pairing age with polarity
draw by draw is the whole point, so the default source is the per-draw table
from `normalize_tes.build_draw_polarity`.

`--ancestral-table` is offered as an explicitly approximate alternative: it
applies the correct per-site weight to the row's *marginal* age distribution,
which is cheap -- no new table to build -- and wrong in exactly the way above.
Use it for a quick look, not for a result.

Draws that are excluded
-----------------------
A draw is usable at a row only when it contributes exactly one age interval and
names one of the two observed alleles ancestral. A draw with several mutations
at the site has no single mutation age and no single derived allele; a draw
naming a third base cannot orient the site. Both are dropped from the row's
denominator rather than resolved by a rule, so the weight is conditioned on the
draws that can actually answer the question. `--min-usable-draws` then drops
rows with too few of them.

Output
------
Per sample: the binned mixture (`mass.npy`, `spectrum.tsv`), its total mass, the
exactly-computed weighted mean age, and bin-interpolated quantiles. Mass is
*unnormalized*: it is the expected number of segregating sites at which the
individual carries a derived allele, so samples remain comparable in scale.
`spectrum.tsv` also carries the normalized probability per bin.

Bins are given in age units of the ARG (generations). The first bin is
`[0, --bin-min)` and the last `[--bin-max, inf)`, so no mass is ever discarded:
mutations on terminal branches genuinely have age intervals starting at 0.

Scale note: one pass over a whole-genome VCF dominates the runtime. Per-sample
cost is linear in the number of samples, so `--samples` is worth using. Run one
job per chromosome and gather with `--merge`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .build_draw_polarity import NO_CALL, open_draw_polarity
from .phi_sfs import _open_vcf
from .release_provenance import software_provenance
from .snp_age_store import open_snp_age_store, store_schema


SCHEMA_VERSION = "individual-age-spectrum-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
BASES = "ACGT"
_BASE_INDEX = {base: index for index, base in enumerate(BASES)}
PROGRESS_RECORDS = 1_000_000
MAX_PLOIDY = 15

# Default binning, in ARG generations: 100-generation resolution across the first
# 10,000 generations, then progressively coarser segments so the deep tail does
# not cost hundreds of near-empty columns. An unbounded bin closes the series.
DEFAULT_BIN_STEPS = (
    "100:10000",
    "1000:100000",
    "5000:500000",
    "10000:2000000",
    "100000:10000000",
)


# ---------------------------------------------------------------- age binning


@dataclass(frozen=True)
class AgeBinning:
    """Bin edges plus the widths used for fully covered interior bins.

    `widths` zeroes the unbounded final bin. An interval only ever contributes
    to interior bins strictly between the bins holding its two endpoints, so
    the final bin is never an interior bin and its width is never needed; a
    literal infinity there would turn an exact zero into a NaN.
    """

    edges: np.ndarray
    widths: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.edges.size - 1)

    @classmethod
    def _finish(cls, inner: np.ndarray) -> "AgeBinning":
        edges = np.asarray(inner, dtype=np.float64)
        if edges[0] != 0.0:
            edges = np.concatenate(([0.0], edges))
        edges = np.concatenate((edges, [np.inf]))
        widths = np.diff(edges)
        widths[-1] = 0.0
        return cls(edges=edges, widths=widths)

    @classmethod
    def build(cls, scale: str, low: float, high: float, count: int) -> "AgeBinning":
        if not 0 < low < high or not math.isfinite(high):
            raise SystemExit("require 0 < --bin-min < --bin-max < inf")
        if count < 1:
            raise SystemExit("--n-bins must be positive")
        if scale == "log":
            inner = np.geomspace(low, high, count + 1)
        elif scale == "linear":
            inner = np.linspace(low, high, count + 1)
        else:
            raise SystemExit("--bin-scale must be 'log' or 'linear'")
        return cls._finish(inner)

    @classmethod
    def from_steps(cls, steps: Sequence[str]) -> "AgeBinning":
        """Build piecewise-constant bins from `WIDTH:LIMIT` segments.

        Resolution should follow where the question is, not a single functional
        form: fine bins where recent mutations land, coarse ones across the deep
        tail that would otherwise cost hundreds of near-empty columns. Segments
        run from 0 through each limit in turn, and an unbounded bin closes the
        series so no mass is discarded.

        A limit that is not a whole number of its own bins is rejected rather
        than absorbed into a short final bin, which would put one silently
        irregular column in the middle of an otherwise regular series.
        """
        if not steps:
            raise SystemExit("--bin-steps needs at least one WIDTH:LIMIT segment")
        edges = [0.0]
        start = 0.0
        for segment in steps:
            width_text, separator, limit_text = segment.partition(":")
            if not separator:
                raise SystemExit(
                    f"--bin-steps segment {segment!r} is not WIDTH:LIMIT")
            try:
                width, limit = float(width_text), float(limit_text)
            except ValueError:
                raise SystemExit(
                    f"--bin-steps segment {segment!r} has a non-numeric value"
                ) from None
            if not math.isfinite(width) or width <= 0:
                raise SystemExit(f"--bin-steps width must be positive: {segment!r}")
            if not math.isfinite(limit) or limit <= start:
                raise SystemExit(
                    f"--bin-steps limits must increase; {limit:g} does not exceed "
                    f"{start:g}")
            span = limit - start
            count = round(span / width)
            if abs(count * width - span) > 1e-9 * max(1.0, limit):
                raise SystemExit(
                    f"--bin-steps segment {segment!r} does not divide evenly: "
                    f"{span:g} generations is not a whole number of {width:g}")
            edges.extend(start + width * np.arange(1, count + 1))
            start = limit
        return cls._finish(np.asarray(edges, dtype=np.float64))


def _bin_mass(row: np.ndarray, below: np.ndarray, above: np.ndarray,
              mass: np.ndarray, binning: AgeBinning, n_rows: int) -> np.ndarray:
    """Integrate uniform interval masses into `(n_rows, n_bins)`.

    Each interval spreads its mass uniformly over `[below, above)`. The two
    end bins are added directly; the fully covered interior run is added as a
    difference array over bins and one cumulative sum, so cost is linear in the
    number of intervals rather than in intervals times bins.
    """
    edges = binning.edges
    n_bins = binning.n_bins
    out = np.zeros((n_rows, n_bins), dtype=np.float64)
    keep = mass > 0
    if not np.any(keep):
        return out
    row = row[keep]
    below = below[keep]
    above = above[keep]
    mass = mass[keep]

    first = np.clip(np.searchsorted(edges, below, side="right") - 1, 0, n_bins - 1)
    last = np.clip(np.searchsorted(edges, above, side="right") - 1, 0, n_bins - 1)
    span = above - below
    single = (span <= 0) | (first == last)

    indices = [row[single] * n_bins + first[single]]
    values = [mass[single]]
    spread = ~single
    if np.any(spread):
        rows = row[spread]
        low_bin = first[spread]
        high_bin = last[spread]
        density = mass[spread] / span[spread]
        indices.append(rows * n_bins + low_bin)
        values.append(density * (edges[low_bin + 1] - below[spread]))
        indices.append(rows * n_bins + high_bin)
        values.append(density * (above[spread] - edges[high_bin]))
        slots = n_bins + 1
        difference = np.bincount(rows * slots + low_bin + 1, weights=density,
                                 minlength=n_rows * slots)
        difference -= np.bincount(rows * slots + high_bin, weights=density,
                                  minlength=n_rows * slots)
        interior = np.cumsum(difference.reshape(n_rows, slots)[:, :n_bins], axis=1)
        out += interior * binning.widths
    out += np.bincount(np.concatenate(indices), weights=np.concatenate(values),
                       minlength=n_rows * n_bins).reshape(n_rows, n_bins)
    return out


# ------------------------------------------------------------- per-row weights


@dataclass
class RowSpectra:
    """One row block's derived-allele age mass, split by which allele is derived."""

    alt_bins: np.ndarray
    ref_bins: np.ndarray
    alt_weight: np.ndarray
    ref_weight: np.ndarray
    alt_mean_numerator: np.ndarray
    ref_mean_numerator: np.ndarray
    valid: np.ndarray


def row_spectra(store, rows: np.ndarray, ref_index: np.ndarray,
                alt_index: np.ndarray, binning: AgeBinning, *,
                polarity: np.ndarray | None,
                marginal_alt: np.ndarray | None,
                marginal_oriented_draws: np.ndarray | None,
                min_usable_draws: int) -> RowSpectra:
    """Split each row's posterior age mass into ALT-derived and REF-derived parts.

    The two parts are kept separate because an individual's genotype selects
    between them: a homozygote takes one, and a heterozygote takes both, which
    is exactly why their sum is the row's whole usable age distribution.
    """
    n_rows = int(rows.size)
    n_draws = int(store.n_posterior_draws)
    batch = store.intervals(rows)
    counts = np.diff(np.asarray(batch.offsets, dtype=np.int64))
    interval_row = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
    below = np.asarray(batch.below, dtype=np.float64)
    above = np.asarray(batch.above, dtype=np.float64)
    draw = np.asarray(batch.draw_id, dtype=np.int64)

    cell = interval_row * n_draws + draw
    per_cell = np.bincount(cell, minlength=n_rows * n_draws)
    # A draw contributing several mutations at one site has neither a single
    # mutation age nor a single derived allele, so it cannot answer the
    # question and is dropped instead of being resolved by a rule.
    single = per_cell[cell] == 1

    if polarity is not None:
        ancestral = np.asarray(polarity[rows]).reshape(-1)[cell]
        alt_derived = single & (ancestral == ref_index[interval_row])
        ref_derived = single & (ancestral == alt_index[interval_row])
        usable = alt_derived | ref_derived
        usable_draws = np.bincount(interval_row[usable], minlength=n_rows)
        valid = usable_draws >= min_usable_draws
        share = np.zeros(n_rows, dtype=np.float64)
        np.divide(1.0, usable_draws, out=share, where=valid)
        alt_mass = np.where(alt_derived & valid[interval_row], share[interval_row], 0.0)
        ref_mass = np.where(ref_derived & valid[interval_row], share[interval_row], 0.0)
    else:
        usable_draws = np.bincount(interval_row[single], minlength=n_rows)
        # The marginal table cannot identify the intersection of draws that
        # both date and orient a row. Still require each marginal separately to
        # meet the requested floor; otherwise a row with many age intervals but
        # only one orienting call would pass --min-usable-draws.
        valid = ((usable_draws >= min_usable_draws)
                 & (marginal_oriented_draws >= min_usable_draws)
                 & np.isfinite(marginal_alt))
        share = np.zeros(n_rows, dtype=np.float64)
        np.divide(1.0, usable_draws, out=share, where=valid)
        base = np.where(single & valid[interval_row], share[interval_row], 0.0)
        proportion = np.where(valid, marginal_alt, 0.0)
        alt_mass = base * proportion[interval_row]
        ref_mass = base * (1.0 - proportion)[interval_row]

    midpoint = 0.5 * (below + above)
    return RowSpectra(
        alt_bins=_bin_mass(interval_row, below, above, alt_mass, binning, n_rows),
        ref_bins=_bin_mass(interval_row, below, above, ref_mass, binning, n_rows),
        alt_weight=np.bincount(interval_row, weights=alt_mass, minlength=n_rows),
        ref_weight=np.bincount(interval_row, weights=ref_mass, minlength=n_rows),
        alt_mean_numerator=np.bincount(interval_row, weights=alt_mass * midpoint,
                                       minlength=n_rows),
        ref_mean_numerator=np.bincount(interval_row, weights=ref_mass * midpoint,
                                       minlength=n_rows),
        valid=valid,
    )


# ----------------------------------------------------------------- VCF reading


class _GenotypeDecoder:
    """Decode GT strings to packed `alt_count * 16 + called_count`.

    Any missing allele makes the individual uncallable at the site rather than
    contributing its called alleles, because a half-called diploid would
    otherwise enter the mixture with a silently halved weight.
    """

    def __init__(self) -> None:
        self.cache: dict[str, int] = {}

    def __call__(self, gt: str) -> int:
        code = self.cache.get(gt)
        if code is None:
            code = self._decode(gt)
            self.cache[gt] = code
        return code

    @staticmethod
    def _decode(gt: str) -> int:
        alleles = gt.replace("|", "/").split("/")
        alt = 0
        called = 0
        for allele in alleles:
            if allele == "." or allele == "":
                return 0
            if not allele.isdigit():
                return -1
            value = int(allele)
            if value > 1:
                return -2
            alt += value
            called += 1
        if called > MAX_PLOIDY:
            return -3
        return alt * 16 + called


@dataclass
class VcfChunk:
    chrom: np.ndarray
    position: np.ndarray
    ref_index: np.ndarray
    alt_index: np.ndarray
    codes: np.ndarray          # (records, samples) uint8


def read_vcf_chunks(path: Path, *, sample_filter: Sequence[str] | None,
                    chunk_records: int, multiallelic: str, progress: bool
                    ) -> Iterator[tuple[list[str], VcfChunk, dict, str]]:
    """Yield sample names then successive record chunks, and finally the digest.

    The digest is accumulated during the same pass rather than in a second full
    read, so a published result names the exact bytes it was computed from.
    """
    handle, hashing, buffered = _open_vcf(path)
    decoder = _GenotypeDecoder()
    stats = {"records": 0, "multiallelic_skipped": 0, "non_acgt_skipped": 0}
    samples: list[str] | None = None
    columns: list[int] = []
    names: list[str] = []
    chrom_buf: list[str] = []
    pos_buf: list[int] = []
    ref_buf: list[int] = []
    alt_buf: list[int] = []
    code_buf: list[list[int]] = []
    next_report = PROGRESS_RECORDS

    def flush() -> VcfChunk:
        chunk = VcfChunk(
            chrom=np.array(chrom_buf, dtype=object),
            position=np.array(pos_buf, dtype=np.int64),
            ref_index=np.array(ref_buf, dtype=np.int16),
            alt_index=np.array(alt_buf, dtype=np.int16),
            codes=np.array(code_buf, dtype=np.uint8).reshape(len(code_buf), len(names)),
        )
        chrom_buf.clear(); pos_buf.clear(); ref_buf.clear()
        alt_buf.clear(); code_buf.clear()
        return chunk

    try:
        for line_number, raw in enumerate(handle, 1):
            if raw.startswith("#"):
                if raw.startswith("#CHROM"):
                    header = raw.rstrip("\n").split("\t")
                    if len(header) < 10:
                        raise SystemExit(f"{path}: header declares no samples")
                    samples = header[9:]
                    if sample_filter is None:
                        columns = list(range(len(samples)))
                    else:
                        lookup = {name: index for index, name in enumerate(samples)}
                        missing = [s for s in sample_filter if s not in lookup]
                        if missing:
                            raise SystemExit(
                                f"{path}: requested samples are absent: "
                                + ", ".join(missing[:5]))
                        columns = [lookup[s] for s in sample_filter]
                    names = [samples[i] for i in columns]
                    yield names, None, None, None
                continue
            if samples is None:
                raise SystemExit(f"{path}:{line_number}: record before #CHROM header")
            stats["records"] += 1
            if progress and stats["records"] >= next_report:
                print(f"  scanned {stats['records']:,} records of {path.name}",
                      flush=True)
                next_report += PROGRESS_RECORDS
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 10:
                raise SystemExit(f"{path}:{line_number}: malformed VCF record")
            try:
                position = int(fields[1])
            except ValueError:
                raise SystemExit(f"{path}:{line_number}: invalid POS") from None
            ref, alt = fields[3].upper(), fields[4].upper()
            if "," in alt:
                if multiallelic == "error":
                    raise SystemExit(
                        f"{path}:{line_number}: multiallelic record at "
                        f"{fields[0]}:{position}; split it upstream or pass "
                        "--multiallelic skip")
                stats["multiallelic_skipped"] += 1
                continue
            if ref not in _BASE_INDEX or alt not in _BASE_INDEX:
                stats["non_acgt_skipped"] += 1
                continue
            format_fields = fields[8].split(":")
            if "GT" not in format_fields:
                raise SystemExit(
                    f"{path}:{line_number}: record lacks GT at {fields[0]}:{position}")
            gt_index = format_fields.index("GT")
            record: list[int] = []
            for column in columns:
                sample = fields[9 + column]
                if gt_index:
                    parts = sample.split(":")
                    gt = parts[gt_index] if gt_index < len(parts) else "."
                else:
                    end = sample.find(":")
                    gt = sample if end < 0 else sample[:end]
                code = decoder(gt)
                if code < 0:
                    raise SystemExit(
                        f"{path}:{line_number}: unusable genotype {gt!r} at "
                        f"{fields[0]}:{position} ("
                        + {-1: "non-numeric allele", -2: "allele index above 1 in a "
                           "biallelic record", -3: "ploidy above "
                           f"{MAX_PLOIDY}"}[code] + ")")
                record.append(code)
            chrom_buf.append(fields[0])
            pos_buf.append(position)
            ref_buf.append(_BASE_INDEX[ref])
            alt_buf.append(_BASE_INDEX[alt])
            code_buf.append(record)
            if len(code_buf) >= chunk_records:
                yield names, flush(), None, None
        if code_buf:
            yield names, flush(), None, None
        while buffered.read(1 << 20):
            pass
    finally:
        handle.close()
    if progress:
        print(f"  scanned {stats['records']:,} records of {path.name}", flush=True)
    yield names, None, stats, hashing.digest.hexdigest()


# -------------------------------------------------------------- accumulation


@dataclass
class Accumulator:
    samples: list[str]
    binning: AgeBinning
    mass: np.ndarray
    weight: np.ndarray
    mean_numerator: np.ndarray
    sites_used: np.ndarray

    @classmethod
    def empty(cls, samples: list[str], binning: AgeBinning) -> "Accumulator":
        n = len(samples)
        return cls(
            samples=samples,
            binning=binning,
            mass=np.zeros((n, binning.n_bins), dtype=np.float64),
            weight=np.zeros(n, dtype=np.float64),
            mean_numerator=np.zeros(n, dtype=np.float64),
            sites_used=np.zeros(n, dtype=np.int64),
        )

    def add(self, spectra: RowSpectra, codes: np.ndarray, dosage: bool) -> None:
        """Fold one row block into every sample's running mixture.

        `codes` packs the called allele count and the ALT count per sample and
        row. A sample's mask is its count of ALT copies and of REF copies (or
        their indicators), which is what makes a heterozygote take both halves
        of the row's mass and a homozygote exactly one.
        """
        alt_count = (codes >> 4).astype(np.float64)
        called = (codes & 15).astype(np.float64)
        ref_count = called - alt_count
        usable = (called > 0) & spectra.valid[None, :]
        if dosage:
            alt_mask = np.where(usable, alt_count, 0.0)
            ref_mask = np.where(usable, ref_count, 0.0)
        else:
            alt_mask = np.where(usable & (alt_count > 0), 1.0, 0.0)
            ref_mask = np.where(usable & (ref_count > 0), 1.0, 0.0)
        self.mass += alt_mask @ spectra.alt_bins + ref_mask @ spectra.ref_bins
        self.weight += alt_mask @ spectra.alt_weight + ref_mask @ spectra.ref_weight
        self.mean_numerator += (alt_mask @ spectra.alt_mean_numerator
                                + ref_mask @ spectra.ref_mean_numerator)
        self.sites_used += usable.sum(axis=1)


def quantiles(mass: np.ndarray, edges: np.ndarray,
              probabilities: Sequence[float]) -> list[float]:
    """Interpolate quantiles inside bins, treating each bin as uniform.

    The unbounded final bin has no interior to interpolate, so a quantile that
    falls in it is reported at its lower edge and is a lower bound.
    """
    total = mass.sum()
    if total <= 0:
        return [float("nan")] * len(probabilities)
    cumulative = np.concatenate(([0.0], np.cumsum(mass) / total))
    output = []
    for probability in probabilities:
        index = int(np.searchsorted(cumulative, probability, side="left"))
        index = min(max(index, 1), mass.size)
        lower, upper = cumulative[index - 1], cumulative[index]
        left, right = edges[index - 1], edges[index]
        if upper <= lower or not math.isfinite(right):
            output.append(float(left))
            continue
        output.append(float(left + (right - left) * (probability - lower) / (upper - lower)))
    return output


# ------------------------------------------------------------------ publishing


def _write(output: Path, accumulator: Accumulator, metadata: dict) -> None:
    if output.exists():
        raise SystemExit(
            f"output already exists: {output}. Refusing to overwrite a published "
            "result; remove it explicitly or choose another path.")
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        binning = accumulator.binning
        arrays = {
            "bin_edges": binning.edges,
            "mass": accumulator.mass,
            "total_weight": accumulator.weight,
            "mean_numerator": accumulator.mean_numerator,
            "sites_used": accumulator.sites_used,
        }
        for name, array in arrays.items():
            with (staging / f"{name}.npy").open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
        (staging / "samples.txt").write_text(
            "\n".join(accumulator.samples) + "\n", encoding="utf-8")

        with (staging / "spectrum.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("sample", "bin_index", "age_low", "age_high",
                             "mass", "density", "probability"))
            for index, sample in enumerate(accumulator.samples):
                row_mass = accumulator.mass[index]
                total = row_mass.sum()
                for b in range(binning.n_bins):
                    low, high = binning.edges[b], binning.edges[b + 1]
                    width = high - low
                    density = row_mass[b] / width if math.isfinite(width) and width > 0 else float("nan")
                    writer.writerow((
                        sample, b, f"{low:.6g}", f"{high:.6g}",
                        f"{row_mass[b]:.10g}", f"{density:.10g}",
                        f"{row_mass[b] / total:.10g}" if total > 0 else "nan",
                    ))

        probabilities = (0.05, 0.25, 0.5, 0.75, 0.95)
        with (staging / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("sample", "sites_used", "total_weight",
                             "mean_age", *(f"q{int(p * 100):02d}_age"
                                           for p in probabilities)))
            for index, sample in enumerate(accumulator.samples):
                weight = accumulator.weight[index]
                mean = accumulator.mean_numerator[index] / weight if weight > 0 else float("nan")
                values = quantiles(accumulator.mass[index], binning.edges, probabilities)
                writer.writerow((
                    sample, int(accumulator.sites_used[index]),
                    f"{weight:.10g}", f"{mean:.10g}",
                    *(f"{v:.10g}" for v in values),
                ))

        (staging / "metadata.json").write_text(
            json.dumps({**metadata, "complete": True}, indent=2, sort_keys=True)
            + "\n", encoding="utf-8")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# ------------------------------------------------------------------------ CLI


def _read_sample_list(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            names.append(text)
    if not names:
        raise SystemExit(f"{path}: no sample names")
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: duplicate sample names")
    return names


def _read_positions(path: Path) -> set[tuple[str, int]]:
    keep: set[tuple[str, int]] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 2:
            raise SystemExit(f"{path}:{number}: expected 'chrom position'")
        try:
            keep.add((parts[0], int(parts[1])))
        except ValueError:
            raise SystemExit(f"{path}:{number}: invalid position") from None
    if not keep:
        raise SystemExit(f"{path}: no positions")
    return keep


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True,
                        help="interval store supplying posterior SNP ages")
    parser.add_argument("--output", type=Path, required=True,
                        help="new result directory")
    parser.add_argument("--vcf", type=Path, nargs="+", default=[],
                        help="biallelic VCF(s) holding the individuals' genotypes")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--draw-polarity", type=Path,
                        help="per-draw polarity table from build_draw_polarity")
    source.add_argument("--ancestral-table", type=Path,
                        help="marginal ancestral-state table; approximate, see the "
                             "module docstring")
    parser.add_argument("--samples", nargs="+",
                        help="sample names to analyse; default every VCF sample")
    parser.add_argument("--samples-file", type=Path,
                        help="file of sample names, one per line")
    parser.add_argument("--include-positions", type=Path,
                        help="restrict to these 'chrom position' sites")
    parser.add_argument("--exclude-positions", type=Path,
                        help="drop these 'chrom position' sites, e.g. all TE positions")
    parser.add_argument("--bin-scale", choices=("steps", "log", "linear"),
                        default="steps",
                        help="'steps' uses --bin-steps; 'log' and 'linear' use "
                             "--bin-min/--bin-max/--n-bins")
    parser.add_argument("--bin-steps", nargs="+", default=list(DEFAULT_BIN_STEPS),
                        metavar="WIDTH:LIMIT",
                        help="piecewise bin widths in generations, each segment "
                             "running to its limit; default "
                             + " ".join(DEFAULT_BIN_STEPS))
    parser.add_argument("--bin-min", type=float, default=1.0)
    parser.add_argument("--bin-max", type=float, default=1e8)
    parser.add_argument("--n-bins", type=int, default=160)
    parser.add_argument("--min-usable-draws", type=int, default=8,
                        help="minimum draws that both date and orient a site")
    parser.add_argument("--allele-weighting", choices=("site", "dosage"),
                        default="site",
                        help="'site' counts a carried derived allele once per site; "
                             "'dosage' counts each carried copy")
    parser.add_argument("--unknown-chromosome", choices=("error", "skip"),
                        default="error",
                        help="VCF chromosome labels absent from the store")
    parser.add_argument("--multiallelic", choices=("error", "skip"), default="error")
    parser.add_argument("--chunk-records", type=int, default=20000,
                        help="VCF records held in memory per accumulation block")
    parser.add_argument("--merge", type=Path, nargs="+",
                        help="part directories to sum into --output")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _merge(args: argparse.Namespace) -> tuple[Accumulator, dict]:
    """Sum part results after checking they describe the same analysis."""
    accumulator: Accumulator | None = None
    detail: list[dict] = []
    seen_paths: set[Path] = set()
    seen_vcfs: dict[str, str] = {}
    seen_digests: dict[str, str] = {}
    reference: dict | None = None
    for part in args.merge:
        resolved = part.resolve()
        if resolved in seen_paths:
            raise SystemExit(f"--merge lists {part} more than once")
        seen_paths.add(resolved)
        try:
            meta = json.loads((part / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"{part}: unreadable part metadata") from error
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise SystemExit(f"{part}: not an individual age spectrum")
        if not meta.get("complete"):
            raise SystemExit(f"{part}: result is incomplete")
        if "merged" in meta:
            raise SystemExit(
                f"{part}: already a merged result; summing it again would "
                "double-count every VCF it covers")
        samples = (part / "samples.txt").read_text(encoding="utf-8").split()
        edges = np.load(part / "bin_edges.npy", allow_pickle=False)
        keys = ("store_content_sha256", "polarity_source", "polarity_table",
                "allele_weighting", "min_usable_draws")
        if reference is None:
            reference = {key: meta.get(key) for key in keys}
            reference["samples"] = samples
            reference["edges"] = edges
            accumulator = Accumulator.empty(
                samples, AgeBinning(edges=edges, widths=_widths(edges)))
        else:
            for key in keys:
                if meta.get(key) != reference[key]:
                    raise SystemExit(f"{part}: {key} differs from the first part")
            if samples != reference["samples"]:
                raise SystemExit(f"{part}: sample set or order differs")
            if not np.array_equal(edges, reference["edges"]):
                raise SystemExit(f"{part}: bin edges differ")
        vcf_entries = meta.get("vcfs", [])
        if not vcf_entries:
            raise SystemExit(f"{part}: records no input VCF")
        try:
            vcfs = {str(entry["path"]): str(entry["sha256"])
                    for entry in vcf_entries}
        except (KeyError, TypeError):
            raise SystemExit(
                f"{part}: each input VCF must record path and sha256") from None
        # Require a real digest rather than merely a non-empty field: a null
        # sha256 stringifies to "None", which passes an emptiness test and
        # would let an unidentifiable input through the checks below.
        bad = [path for path, digest in vcfs.items()
               if not path or not _SHA256.fullmatch(digest)]
        if bad:
            raise SystemExit(
                f"{part}: input VCF metadata lacks a usable path and sha256: "
                + ", ".join(sorted(bad)[:3]))
        if len(vcfs) != len(vcf_entries):
            raise SystemExit(f"{part}: records the same VCF path more than once")
        if len(set(vcfs.values())) != len(vcfs):
            raise SystemExit(
                f"{part}: records one file under two paths; merging it would "
                "count those sites twice")
        # Two checks, because a path and its contents fail in different ways.
        # The same path in two parts is a repeated input. The same *bytes* under
        # two paths -- a symlink, a copy, an absolute and a relative spelling --
        # is the same input wearing a disguise, and summing it would double every
        # site it holds. Neither is visible to the other check.
        repeated = sorted(set(seen_vcfs).intersection(vcfs))
        if repeated:
            changed = [path for path in repeated if seen_vcfs[path] != vcfs[path]]
            if changed:
                raise SystemExit(
                    f"{part}: {changed[0]} was already gathered with different "
                    "contents; the file changed between the runs being merged")
            raise SystemExit(
                f"{part}: VCFs already counted by an earlier part: "
                + ", ".join(repeated[:3]))
        aliased = sorted((path, seen_digests[digest])
                         for path, digest in vcfs.items() if digest in seen_digests)
        if aliased:
            here, before = aliased[0]
            raise SystemExit(
                f"{part}: {here} holds the same bytes as {before}, already "
                "gathered from an earlier part; merging both would count those "
                "sites twice")
        seen_vcfs.update(vcfs)
        seen_digests.update({digest: path for path, digest in vcfs.items()})
        accumulator.mass += np.load(part / "mass.npy", allow_pickle=False)
        accumulator.weight += np.load(part / "total_weight.npy", allow_pickle=False)
        accumulator.mean_numerator += np.load(part / "mean_numerator.npy",
                                              allow_pickle=False)
        accumulator.sites_used += np.load(part / "sites_used.npy", allow_pickle=False)
        detail.append({"path": str(resolved), "vcfs": vcf_entries})
    report = {**{k: reference[k] for k in
                 ("store_content_sha256", "polarity_source", "polarity_table",
                  "allele_weighting", "min_usable_draws")},
              "merged": detail, "merged_parts": len(detail),
              "vcfs": [{"path": path, "sha256": seen_vcfs[path]}
                       for path in sorted(seen_vcfs)]}
    return accumulator, report


def _widths(edges: np.ndarray) -> np.ndarray:
    widths = np.diff(edges)
    widths[-1] = 0.0
    return widths


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    progress = not args.quiet

    if args.merge:
        if args.vcf:
            raise SystemExit("--vcf cannot be combined with --merge")
        accumulator, report = _merge(args)
        metadata = {"schema_version": SCHEMA_VERSION,
                    "samples": accumulator.samples,
                    "bins": accumulator.binning.n_bins,
                    "software": software_provenance(), **report}
        _write(args.output, accumulator, metadata)
        print(f"wrote {args.output}")
        return 0

    if not args.vcf:
        raise SystemExit("give at least one --vcf, or use --merge")
    # Check this before the scan, not at publication time: a whole-genome pass
    # costs hours, and refusing afterwards throws all of it away.
    if args.output.exists():
        raise SystemExit(
            f"output already exists: {args.output}. Refusing to overwrite a "
            "published result; remove it explicitly or choose another path.")
    if args.samples and args.samples_file:
        raise SystemExit("give --samples or --samples-file, not both")
    if not args.draw_polarity and not args.ancestral_table:
        raise SystemExit(
            "give --draw-polarity for the per-draw analysis, or --ancestral-table "
            "for the approximate marginal one")
    if args.min_usable_draws < 1:
        raise SystemExit("--min-usable-draws must be positive")
    if args.chunk_records < 1:
        raise SystemExit("--chunk-records must be positive")

    sample_filter = args.samples or (
        _read_sample_list(args.samples_file) if args.samples_file else None)
    include = _read_positions(args.include_positions) if args.include_positions else None
    exclude = _read_positions(args.exclude_positions) if args.exclude_positions else None
    binning = (AgeBinning.from_steps(args.bin_steps) if args.bin_scale == "steps"
               else AgeBinning.build(args.bin_scale, args.bin_min, args.bin_max,
                                     args.n_bins))

    store = open_snp_age_store(args.store)
    store_metadata = getattr(store, "metadata", {}) or {}
    positions = np.asarray(store.positions)
    offsets = {str(entry["chrom"]): int(entry["offset"])
               for entry in store_metadata.get("chromosomes", [])}
    if not offsets:
        raise SystemExit("the interval store records no chromosome offsets")

    polarity_table = None
    marginal = None
    polarity_metadata: dict = {}
    if args.draw_polarity:
        polarity_table, polarity_metadata = open_draw_polarity(args.draw_polarity, store)
    else:
        table = Path(args.ancestral_table)
        try:
            polarity_metadata = json.loads(
                (table / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"{table}: unreadable ancestral-table metadata") from error
        if polarity_metadata.get("store_content_sha256") != store_metadata.get("content_sha256"):
            raise SystemExit(
                f"{table}: built from a different interval store than --store")
        counts = np.load(table / "ancestral_counts.npy", mmap_mode="r",
                         allow_pickle=False)
        if counts.shape != (positions.size, 4):
            raise SystemExit(
                f"{table}: ancestral_counts.npy has shape {counts.shape}, expected "
                f"{(positions.size, 4)}")
        marginal = counts

    accumulator = Accumulator.empty([], binning)
    established: list[str] | None = None
    stats = {"records": 0, "unresolved": 0, "excluded": 0, "not_included": 0,
             "unknown_chromosome": 0, "multiallelic_skipped": 0,
             "non_acgt_skipped": 0, "rows_below_min_usable_draws": 0,
             "rows_used": 0}
    vcf_report: list[dict] = []

    for vcf in args.vcf:
        if progress:
            print(f"reading {vcf}", flush=True)
        digest = None
        for names, chunk, chunk_stats, chunk_digest in read_vcf_chunks(
                vcf, sample_filter=sample_filter, chunk_records=args.chunk_records,
                multiallelic=args.multiallelic, progress=progress):
            if established is None:
                established = names
                accumulator = Accumulator.empty(names, binning)
            elif names != established:
                raise SystemExit(
                    f"{vcf}: sample set or order differs from the first VCF; pass "
                    "--samples to fix an explicit order")
            if chunk_stats is not None:
                stats["records"] += chunk_stats["records"]
                stats["multiallelic_skipped"] += chunk_stats["multiallelic_skipped"]
                stats["non_acgt_skipped"] += chunk_stats["non_acgt_skipped"]
                digest = chunk_digest
                continue
            if chunk is None:
                continue
            _accumulate_chunk(chunk, store, positions, offsets, binning,
                              polarity_table, marginal, accumulator, args,
                              include, exclude, stats)
        vcf_report.append({"path": str(Path(vcf).resolve()), "sha256": digest})

    if established is None:
        raise SystemExit("no VCF declared any samples")
    if stats["rows_used"] == 0:
        if stats["rows_below_min_usable_draws"]:
            raise SystemExit(
                f"all {stats['rows_below_min_usable_draws']:,} resolved sites fell "
                f"below --min-usable-draws {args.min_usable_draws}; the store or the "
                "polarity table gives them too few draws that both date and orient "
                "the site")
        raise SystemExit(
            "no VCF site resolved to a store row; check that the VCF chromosome "
            "labels and coordinates match the store's")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "store": str(Path(args.store).resolve()),
        "store_schema": store_schema(store),
        "store_content_sha256": store_metadata.get("content_sha256"),
        "store_rows": int(positions.size),
        "posterior_draws": int(store.n_posterior_draws),
        "polarity_source": "per-draw" if args.draw_polarity else "marginal-approximate",
        "polarity_table": str(Path(args.draw_polarity or args.ancestral_table).resolve()),
        "polarity_table_schema": polarity_metadata.get("schema_version"),
        "allele_weighting": args.allele_weighting,
        "min_usable_draws": args.min_usable_draws,
        "bin_scale": args.bin_scale,
        "bin_steps": list(args.bin_steps) if args.bin_scale == "steps" else None,
        "bin_min": None if args.bin_scale == "steps" else args.bin_min,
        "bin_max": None if args.bin_scale == "steps" else args.bin_max,
        "n_inner_bins": None if args.bin_scale == "steps" else args.n_bins,
        "bins": binning.n_bins,
        "samples": established,
        "vcfs": vcf_report,
        "counts": stats,
        "mass_units": "expected segregating sites carrying a derived allele",
        "software": software_provenance(),
    }
    _write(args.output, accumulator, metadata)

    print(f"samples            {len(established)}")
    print(f"VCF records        {stats['records']:,}")
    print(f"store rows used    {stats['rows_used']:,}")
    print(f"unresolved sites   {stats['unresolved']:,}")
    print(f"below draw floor   {stats['rows_below_min_usable_draws']:,}")
    print(f"wrote {args.output}")
    return 0


def _accumulate_chunk(chunk: VcfChunk, store, positions: np.ndarray,
                      offsets: dict[str, int], binning: AgeBinning,
                      polarity_table, marginal, accumulator: Accumulator,
                      args: argparse.Namespace, include, exclude,
                      stats: dict) -> None:
    """Resolve one VCF block to store rows and fold it into the accumulator."""
    keep = np.ones(chunk.position.size, dtype=bool)
    if include is not None or exclude is not None:
        for index, (chrom, position) in enumerate(zip(chunk.chrom, chunk.position)):
            coordinate = (str(chrom), int(position))
            if include is not None and coordinate not in include:
                keep[index] = False
                stats["not_included"] += 1
            elif exclude is not None and coordinate in exclude:
                keep[index] = False
                stats["excluded"] += 1
    # A VCF block almost always sits on one chromosome, so resolve the offset
    # per distinct label and broadcast it rather than per record.
    global_positions = np.full(chunk.position.size, -1.0, dtype=np.float64)
    labels = np.asarray([str(value) for value in chunk.chrom], dtype=object)
    for label in set(labels.tolist()):
        where = labels == label
        offset = offsets.get(label)
        if offset is None:
            if args.unknown_chromosome == "error":
                raise SystemExit(
                    f"VCF chromosome {label!r} is absent from the store; known "
                    "labels are " + ", ".join(sorted(offsets)[:8]))
            stats["unknown_chromosome"] += int(np.count_nonzero(where & keep))
            keep &= ~where
            continue
        global_positions[where] = offset + chunk.position[where]

    selection = np.flatnonzero(keep)
    if selection.size == 0:
        return
    targets = global_positions[selection]
    insertion = np.searchsorted(positions, targets)
    found = insertion < positions.size
    found[found] &= positions[insertion[found]] == targets[found]
    stats["unresolved"] += int(np.count_nonzero(~found))
    selection = selection[found]
    rows = insertion[found]
    if rows.size == 0:
        return
    if np.unique(rows).size != rows.size:
        raise SystemExit(
            "two VCF records resolve to the same store row inside one block; the "
            "VCF holds duplicate coordinates")

    ref_index = chunk.ref_index[selection].astype(np.int64)
    alt_index = chunk.alt_index[selection].astype(np.int64)
    marginal_alt = None
    marginal_oriented_draws = None
    if marginal is not None:
        counts = np.asarray(marginal[rows], dtype=np.float64)
        take = np.arange(rows.size)
        ref_calls = counts[take, ref_index]
        alt_calls = counts[take, alt_index]
        oriented = ref_calls + alt_calls
        # ALT is derived exactly when REF is the ancestral call, conditioned on
        # the draws that named one of the two observed alleles.
        marginal_alt = np.where(oriented > 0, ref_calls / np.maximum(oriented, 1),
                                np.nan)
        marginal_oriented_draws = oriented

    spectra = row_spectra(store, rows, ref_index, alt_index, binning,
                          polarity=polarity_table, marginal_alt=marginal_alt,
                          marginal_oriented_draws=marginal_oriented_draws,
                          min_usable_draws=args.min_usable_draws)
    stats["rows_below_min_usable_draws"] += int(np.count_nonzero(~spectra.valid))
    stats["rows_used"] += int(np.count_nonzero(spectra.valid))
    accumulator.add(spectra, chunk.codes[selection].T,
                    args.allele_weighting == "dosage")


if __name__ == "__main__":
    raise SystemExit(main())
