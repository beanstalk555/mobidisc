import datetime
from collections import Counter
import src.permrep as perm
import src.ranloop as ranloop
import src.drawloop as drawloop
from src.mobidisc_processor import MobidiscProcessor
from src.shd_wrapper import SHDWrapper
import pandas as pd
import re
from ast import literal_eval


def display_to_tuple(display_str: str) -> list[tuple[int]]:
    return [
        tuple(map(int, match.group(1).split(",")))
        for match in re.finditer(r"\(([^)]+)\)", display_str)
    ]


def main():
    loops_data = pd.read_csv("data/loops.txt", sep="\t")
    cnf = []
    for index, row in loops_data.iterrows():
        print(f"Processing loop: {row['name']}")
        sigma_string = row["sigma"]
        sigma_tuple = display_to_tuple(sigma_string)
        loop = perm.Multiloop(sigma_tuple)
        processed_loop = MobidiscProcessor(loop)
        drawloop.DrawLoop(
            sequences=processed_loop.main_sequence,
            circle_dict=processed_loop.packed_circles,
            infCircLabel=processed_loop.loop_to_circles.faces_circles[loop.inf_face[0]],
            filename=f"data/loops_data/{row['name']}.svg",
            showCircLabels=processed_loop.face_circles,
        )
        shd_wrapper = SHDWrapper()
        shd_output = shd_wrapper.find_minimal_hitting_sets(processed_loop.mobidiscs_cnf)
        refinedPin = literal_eval(loops_data.iloc[index]["refinedPinSetMat"])

        output_regions = [frozenset(r) for r in shd_output]
        tested_regions = [
            frozenset(int(x) for x in row[2].strip("{}").split(","))
            for row in refinedPin[1:]
        ]
        cnf.append(
            {
                "id": loops_data.iloc[index]["id"],
                "name": loops_data.iloc[index]["name"],
                "mobidiscs_cnf": processed_loop.mobidiscs_cnf,
                "output_regions": output_regions,
                "tested_regions": tested_regions,
                "correct_regions": Counter(tested_regions) == Counter(output_regions),
            }
        )
    cnf_df = pd.DataFrame(cnf)
    cnf_df.to_csv("data/loops_cnf.csv", index=False)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"Execution time: {duration}")
