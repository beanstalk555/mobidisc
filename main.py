import datetime
import permrep as perm
import ranloop
import drawloop
from mobidisc_processor import MobidiscProcessor
import pandas as pd
import re


def main():
    # loops_data = pd.read_csv("data/loops.txt", sep="\t")

    # for index, row in loops_data.iterrows():
    #     print(f"Processing loop: {row['name']}")
    #     sigma_string = row["sigma"]
    #     sigma_tuple = [
    #         tuple(map(int, match.group(1).split(",")))
    #         for match in re.finditer(r"\(([^)]+)\)", sigma_string)
    #     ]
    #     loop = perm.Multiloop(sigma_tuple)
    #     proccessed_loop = MobidiscProcessor(loop)
    #     drawloop.DrawLoop(
    #         sequences=proccessed_loop.main_sequence,
    #         circle_dict=proccessed_loop.packed_circles,
    #         infCircLabel=proccessed_loop.loop_to_circles.faces_circles[
    #             loop.inf_face[0]
    #         ],
    #         filename=f"data/loops_data/{row['name']}.svg",
    #         showCircLabels=proccessed_loop.face_circles,
    #     )
    loop = perm.Multiloop(
        [
            (10, 1, -11, -2),
            (3, 8, -4, -9),
            (4, 11, -5, -12),
            (12, 5, -1, -6),
            (9, 6, -10, -7),
            (7, 2, -8, -3),
        ],
        (-4, -12, -6, 9),
    )
    proccessed_loop = MobidiscProcessor(loop)
    print(proccessed_loop.loop_to_circles.faces_circles)
    drawloop.DrawLoop(
        sequences=proccessed_loop.main_sequence,
        circle_dict=proccessed_loop.packed_circles,
        infCircLabel=proccessed_loop.loop_to_circles.faces_circles[loop.inf_face[0]],
        filename=f"data/loop.svg",
        showCircLabels=proccessed_loop.face_circles,
    )


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"Execution time: {duration}")
