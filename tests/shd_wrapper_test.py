import sys
import os
import random
import re
import pandas as pd
import unittest
import tempfile
from collections import Counter
from ast import literal_eval
from src.shd_wrapper import SHDWrapper
import src.permrep as perm
from src.mobidisc_processor import MobidiscProcessor


class TestSHDMethods(unittest.TestCase):
    def test_write_input_file_empty(self):
        shd = SHDWrapper()
        cnf_clauses = []
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            shd.write_input_file(cnf_clauses, tmpfile.name)
            tmpfile.close()
            with open(tmpfile.name, "r") as f:
                content = f.read()
                self.assertEqual(content, "")
            os.remove(tmpfile.name)

    def test_write_input_file_single_clause(self):
        shd = SHDWrapper()
        cnf_clauses = [[5]]
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            shd.write_input_file(cnf_clauses, tmpfile.name)
            tmpfile.close()
            with open(tmpfile.name, "r") as f:
                content = f.read()
                self.assertEqual(content, "5\n")
            os.remove(tmpfile.name)

    def test_write_input_file_multiple_clauses(self):
        shd = SHDWrapper()
        cnf_clauses = [[1, 2, 3], [2, 4], [1, 4, 5]]
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            shd.write_input_file(cnf_clauses, tmpfile.name)
            tmpfile.close()
            with open(tmpfile.name, "r") as f:
                content = f.read()
                expected_content = "1 2 3\n2 4\n1 4 5\n"
                self.assertEqual(content, expected_content)
            os.remove(tmpfile.name)

    def test_parse_output_file_empty(self):
        shd = SHDWrapper()
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmpfile:
            tmpfile.write("")
            tmpfile.close()
            result = shd.parse_output_file(tmpfile.name)
            self.assertEqual(result, [])
            os.remove(tmpfile.name)

    def test_parse_output_file_single_solution(self):
        shd = SHDWrapper()
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmpfile:
            tmpfile.write("1 2 3\n")
            tmpfile.close()
            result = shd.parse_output_file(tmpfile.name)
            self.assertEqual(result, [[1, 2, 3]])
            os.remove(tmpfile.name)

    def test_parse_output_file_multiple_solutions(self):
        shd = SHDWrapper()
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmpfile:
            tmpfile.write("1 2\n3 4 5\n2\n")
            tmpfile.close()
            result = shd.parse_output_file(tmpfile.name)
            self.assertEqual(result, [[2], [1, 2], [3, 4, 5]])
            os.remove(tmpfile.name)

    def test_parse_output_file_unsorted_elements(self):
        shd = SHDWrapper()
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmpfile:
            tmpfile.write("3 1 2\n5 4\n")
            tmpfile.close()
            result = shd.parse_output_file(tmpfile.name)
            self.assertEqual(result, [[4, 5], [1, 2, 3]])
            os.remove(tmpfile.name)

    def test_shd_binary_not_found(self):
        with self.assertRaises(FileNotFoundError):
            SHDWrapper(shd_binary_path="/nonexistent/path/to/shd")

    def test_find_minimal_hitting_sets(self):
        loops_data = pd.read_csv("tests/fixtures/test_loops_data.txt", sep="\t")

        sample = random.sample(range(0, len(loops_data)), 5)
        shd = SHDWrapper()

        for i in sample:
            sigma_string = loops_data.iloc[i]["sigma"]
            sigma_tuple = [
                tuple(map(int, match.group(1).split(",")))
                for match in re.finditer(r"\(([^)]+)\)", sigma_string)
            ]
            loop = perm.Multiloop(sigma_tuple)
            processed_loop = MobidiscProcessor(loop)
            cnf_clauses = processed_loop.mobidiscs_cnf

            output_regions = shd.find_minimal_hitting_sets(cnf_clauses)

            refinedPin = literal_eval(loops_data.iloc[i]["refinedPinSetMat"])
            expected_regions = [
                frozenset(int(x) for x in row[2].strip("{}").split(","))
                for row in refinedPin[1:]
            ]

            output_regions = [frozenset(r) for r in output_regions]
            self.assertEqual(
                Counter(output_regions),
                Counter(expected_regions),
                msg=f"Failed for loop index {i} with name {loops_data.iloc[i]['name']}",
            )


if __name__ == "__main__":
    unittest.main()
