import math
import tempfile
import unittest
from pathlib import Path

import tskit
import tszip

from snp_age_distribution import AgeInterval, collect_intervals, discretize_intervals


def make_ts(path: Path) -> None:
    tables = tskit.TableCollection(sequence_length=100)
    child = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    parent = tables.nodes.add_row(time=1200)
    root = tables.nodes.add_row(time=3000)
    tables.edges.add_row(0, 100, parent, child)
    tables.edges.add_row(0, 100, root, parent)
    site = tables.sites.add_row(25, "0")
    tables.mutations.add_row(site, child, "1")
    tables.mutations.add_row(site, parent, "2")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    tables.tree_sequence().dump(path)


class DistributionTests(unittest.TestCase):
    def test_exact_uniform_bin_integration_and_normalization(self):
        interval = AgeInterval(1, 0, 2000, "x", 0)
        observed = discretize_intervals([interval], 1000)
        self.assertEqual(set(observed), {0, 1000, 2000})
        self.assertAlmostEqual(observed[0], 0.25)
        self.assertAlmostEqual(observed[1000], 0.5)
        self.assertAlmostEqual(observed[2000], 0.25)
        self.assertTrue(math.isclose(sum(observed.values()), 1))

    def test_zero_width_is_point_mass(self):
        interval = AgeInterval(1, 1500, 1500, "x", 0)
        self.assertEqual(discretize_intervals([interval]), {2000: 1})

    def test_multiple_mutations_and_missing_site(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "example.trees"
            make_ts(filename)
            intervals, missing = collect_intervals([filename], [25, 30])
            self.assertCountEqual(
                [(x.below, x.above) for x in intervals[25]],
                [(0, 1200), (1200, 3000)],
            )
            self.assertEqual(intervals[30], [])
            self.assertEqual(missing, {25: 0, 30: 1})

    def test_root_mutation_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "root.trees"
            tables = tskit.TableCollection(sequence_length=10)
            root = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=5)
            site = tables.sites.add_row(2, "0")
            tables.mutations.add_row(site, root, "1")
            tables.tree_sequence().dump(filename)
            self.assertEqual(collect_intervals([filename], [2])[0][2], [])
            with self.assertRaisesRegex(ValueError, "root"):
                collect_intervals([filename], [2], root="error")

    def test_tsz_input(self):
        with tempfile.TemporaryDirectory() as directory:
            ordinary = Path(directory) / "example.trees"
            compressed = Path(directory) / "example.tsz"
            make_ts(ordinary)
            tszip.compress(tskit.load(ordinary), compressed)
            intervals, missing = collect_intervals([compressed], [25])
            self.assertEqual(len(intervals[25]), 2)
            self.assertEqual(missing[25], 0)


if __name__ == "__main__":
    unittest.main()
