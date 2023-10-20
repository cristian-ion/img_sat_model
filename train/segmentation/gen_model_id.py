import os
from datetime import datetime


def gen_model_id(namecode: str, major_version: int = 1, out_dir: str = None):
    if out_dir:
        files = os.listdir(out_dir)
        files = [f.split(".")[0] for f in files if f[-3:] == ".pt"]
        versions = [tuple(m.split("_")[-3:]) for m in files]
        versions = [tuple(map(int, v)) for v in versions]
        versions.sort(key=lambda x: (x[0], x[1], x[2]))

        if not versions:
            latest = (-1, 0, 0)
        else:
            latest = versions[-1]

        next_version = (0, 0, 0)
        if latest[0] > major_version:
            print("Please increase the major version constant manually.")
            raise Exception("Please increase the major version constant manually.")
        if latest[0] < major_version:
            print(f"New major version {major_version}")
            next_version = (major_version, 0, 0)
        else:
            minor = latest[1]
            subminor = latest[2] + 1
            if subminor > 9:
                subminor = 0
                minor += 1
            if minor > 9:
                raise Exception(
                    "Minor version > 9, please increase major version constant manually."
                )
            next_version = (latest[0], minor, subminor)

        major_version = next_version[0]
        minor_version = f"{next_version[1]}_{next_version[2]}"
    else:
        minor_version = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    return f"{namecode}_model_{major_version}_{minor_version}"
