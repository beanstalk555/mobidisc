import datetime
import permrep as perm
import ranloop as ranloop
import drawloop as drawloop
from mobidisc_processor import MobidiscProcessor
import pandas as pd
import re

from shd_wrapper import SHDWrapper


def display_to_tuple(display_str: str) -> tuple[tuple[int, ...], ...]:
    return [
        tuple(map(int, match.group(1).split(",")))
        for match in re.finditer(r"\(([^)]+)\)", display_str)
    ]


def main():

    # for index, row in loops_data.iterrows():
    #     print(f"Processing loop: {row['name']}")
    #     sigma_string = row["sigma"]
    #     sigma_tuple = display_to_tuple(sigma_string)
    #     loop = perm.Multiloop(sigma_tuple)
    #     processed_loop = MobidiscProcessor(loop)
    #     drawloop.DrawLoop(
    #         sequences=processed_loop.main_sequence,
    #         circle_dict=processed_loop.packed_circles,
    #         infCircLabel=processed_loop.loop_to_circles.faces_circles[
    #             loop.inf_face[0]
    #         ],
    #         filename=f"data/loops_data/{row['name']}.svg",
    #         showCircLabels=processed_loop.face_circles,
    #     )
    loops_data = pd.read_csv("data/loops.txt", sep="\t")
    loop_index = 474
    loop = perm.Multiloop(display_to_tuple(loops_data.iloc[loop_index]["sigma"]))
    processed_loop = MobidiscProcessor(loop)
    print(processed_loop.loop_to_circles.faces_circles)
    drawloop.DrawLoop(
        sequences=processed_loop.main_sequence,
        circle_dict=processed_loop.packed_circles,
        infCircLabel=processed_loop.loop_to_circles.faces_circles[loop.inf_face[0]],
        filename=f"data/test_loop.svg",
        showCircLabels=processed_loop.face_circles,
    )
    cnf_data = {
        "id": loops_data.iloc[loop_index]["id"],
        "name": loops_data.iloc[loop_index]["name"],
        "mobidiscs_cnf": processed_loop.mobidiscs_cnf,
    }
    shd_wrapper = SHDWrapper()
    print(cnf_data["name"])
    print(shd_wrapper.find_minimal_hitting_sets(cnf_data["mobidiscs_cnf"]))
    print("Tested dnf:", loops_data.iloc[loop_index]["refinedPinSetMat"])


import subprocess

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"Execution time: {duration}")