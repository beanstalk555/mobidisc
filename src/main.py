import datetime
import permrep as perm
import ranloop as ranloop
import drawloop as drawloop
from mobidisc_processor import MobidiscProcessor
import pandas as pd
import re


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
    loop_index = 0
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


import subprocess

if __name__ == "__main__":
    # start_time = datetime.datetime.now()
    # main()
    # end_time = datetime.datetime.now()
    # duration = end_time - start_time
    # print(f"Execution time: {duration}")
    try:
        result = subprocess.run(
            ["wsl", "./bin/shd/linux_x86_64/shd", "0", "./data/test.dat"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("EXIT CODE:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
