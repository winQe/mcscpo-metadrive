import random

import matplotlib.pyplot as plt

from metadrive import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map


if __name__ == "__main__":
    print("Drawing current map")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)  # Adjust the size as needed

    map_config = {
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
        BaseMap.GENERATE_CONFIG: "X",  # 3 block
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 2,
    }

    map_config["config"]="SXCOY"

    env = SafeMetaDriveEnv(dict(map_config = map_config))
    env.reset()
    m = draw_top_down_map(env.current_map)
    ax.imshow(m, cmap="bone")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    env.close()
