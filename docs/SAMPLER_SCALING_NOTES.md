# Sampler scaling notes

Recorded 2026-08-14 for follow-up work on running age-matched controls for
many TE categories.

## Observed production run

- SLURM job `37202969` ran `normalize_tes/sample_age_matched_syn.py` from the external
  `sampling.sh` script.
- The job used the RNA-structure TE target, requested 100 accepted sets, and
  allowed up to 200,000 proposals.
- The TE category contained 35,512 requested positions.
- The candidate argument was
  `run.combined.99.age.max6mis.snp.pos.txt`, containing 23,359,072 positions.
- After roughly 19 hours the job was still running with empty stdout and no
  result directory. The job had a 24-hour time limit.
- Empty output does not identify a hang: the sampler has no progress logging
  and publishes its output atomically only after all requested sets succeed.
- The preceding target/bootstrap job completed in 4 minutes 24 seconds and
  reported a Wasserstein acceptance threshold of 2,082.03 generations.

## Diagnosis

The scalar Wasserstein calculation is unlikely to be the main cost. It is a
vectorized sum over the age grid. Wasserstein filtering can nevertheless
increase total runtime indirectly when the acceptance rate is low and many
proposals must be generated.

The dominant scaling problems are more likely to be:

1. Constructing target-specific boundary weights for 23.36 million candidate
   SNPs from the 75-draw interval store.
2. For every proposal and every age stratum, copying and scanning a probability
   vector covering the complete candidate pool before calling weighted
   `rng.choice`.
3. Reading selected candidates from the main Quobyte interval store again to
   construct and score every proposed full CDF.
4. Repeating candidate resolution, interval access, and setup independently
   for every TE category.

The current four-CPU allocation offers little benefit because the principal
sampling path is effectively single-threaded.

## First scientific/input check

Confirm whether the 23.36-million-position file is intentionally the eligible
control pool. The command is named `normalize_tes/sample_age_matched_syn.py`, but the active
argument appears to contain all SNPs. A previously benchmarked synonymous list
contained approximately 485,000 positions. If that smaller list is the intended
control pool, using it would reduce candidate-dependent work by roughly 48-fold
and is the highest-priority correction.

## Recommended implementation order

1. Use the scientifically intended synonymous/control candidate list.
2. Resolve the candidate coordinates once, save the row indices, and reuse
   them with `--syn-indices` for every category.
3. Build a persistent candidate-only interval cache once and reuse it across
   categories. Prefer node-local scratch during a batch when practical.
4. Keep that cache available during proposal scoring instead of returning to
   the full Quobyte store for each proposed set.
5. Replace repeated full-vector weighted sampling with prebuilt weighted alias
   tables, one per stratum. Draw from the tables and reject/redraw candidates
   already selected in the current set. This preserves sequential weighted
   sampling semantics while making proposal cost depend mainly on the selected
   set size rather than the complete candidate count.
6. Add flushed progress and timing records for candidate loading/resolution,
   weight construction, proposals attempted, sets accepted, acceptance rate,
   and scoring time.
7. Add resumable or periodic diagnostic output so a time-limited job provides
   useful acceptance information even if it does not finish.
8. Run a pilot with one accepted set and a small proposal limit before each
   production campaign. Use it to distinguish setup time from proposal time
   and to estimate the acceptance rate.
9. Once the single-category path is efficient, run independent categories as
   a one-CPU SLURM job array. Avoid launching many jobs that independently scan
   the complete remote store at the same time.

## Intended reusable workflow

```text
all-draw interval store
        |
        | one-time preparation
        v
resolved control indices + compact candidate interval cache
        |
        | reused by every category
        v
category target -> alias-based proposals -> full-CDF scoring -> W1 acceptance
```

## Measurements needed before production

For at least one representative small and large TE category, record:

- eligible candidate count and number of target strata;
- time to resolve candidates;
- time and memory to construct boundary weights/alias tables;
- time per proposal;
- time spent retrieving intervals and scoring the full CDF;
- proposals attempted, accepted sets, and acceptance rate;
- rejection reasons, separating Wasserstein rejection from sampling failure;
- candidate-cache size and remote versus node-local read time; and
- peak RSS and effective CPU utilization.

Do not respond to poor acceptance merely by increasing `--max-proposals`.
First confirm the candidate pool has usable mass across the target age range
and inspect the observed Wasserstein-distance distribution.
