"""
LICENSE.

This project as a whole is licensed under the Apache License, Version 2.0.

THIRD-PARTY LICENSES

Third-party software already included in HoloAgent is governed by the separate
Open Source license terms under which the third-party software has been
distributed.

NOTICE ON LICENSE COMPATIBILITY FOR DISTRIBUTORS

Notably, this project depends on the third-party software FAST-LIVO2 and HOVSG.
Their default licenses restrict commercial use—separate permission from their
original authors is required for commercial integration/redistribution.

The third-party software FAST-LIVO2 dependency (licensed under GPL-2.0-only)
utilizes rpg_vikit-ros2 which contains components under the GPL-3.0. Please be
aware of license compatibility when distributing a combined work.

DISCLAIMER

Users are solely responsible for ensuring compliance with all applicable
license terms when using, modifying, or distributing the project. Project
maintainers accept no liability for any license violations arising from such
use.
"""
import os
import sys

import hydra
from omegaconf import DictConfig

from memory.hmsg.graph.graph import Graph

# pylint: disable=all

scene_ids = [
    # "00824-Dd4bFSTQ8gi",
    # "00829-QaLdnwvtxbs",
    # "00843-DYehNKdT76V",
    # "00861-GLAQ4DNUx5U",
    # "00862-LT9Jq6dN3Ea",
    "00873-bxsVRursffK",
    # "00877-4ok3usBNeis",
    # "00890-6s7QHgap2fW",
]

skip_frames_dict = dict()
skip_frames_dict["00824-Dd4bFSTQ8gi"] = 10
skip_frames_dict["00829-QaLdnwvtxbs"] = 10
skip_frames_dict["00843-DYehNKdT76V"] = 10
skip_frames_dict["00861-GLAQ4DNUx5U"] = 16
skip_frames_dict["00862-LT9Jq6dN3Ea"] = 30
skip_frames_dict["00873-bxsVRursffK"] = 20
skip_frames_dict["00877-4ok3usBNeis"] = 15
skip_frames_dict["00890-6s7QHgap2fW"] = 15


@hydra.main(version_base=None, config_path="../../config",
            config_name="semantic_scene_reconstruction_hm3d")
def main(params: DictConfig):

    for scene_id in scene_ids:
        params.main.scene_id = scene_id
        # params.main.scene_id = scene_id
        # Create save directory
        dataset_path = params.main.dataset_path
        params.main.dataset_path = os.path.join(
            params.main.dataset_path, scene_id)  # params.main.scene_id
        params.pipeline.skip_frames = skip_frames_dict[scene_id]
        save_path = params.main.save_path
        save_dir = os.path.join(
            params.main.save_path,
            params.main.dataset,
            scene_id)  # params.main.scene_id
        params.main.save_path = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        print("dataset_path: ", params.main.dataset_path)
        print("save_path: ", save_dir)
        # Create graph
        hmsg = Graph(params)

        # Create feature map
        hmsg.create_feature_map()

        # Save full point cloud, features, and masked point clouds (pcd for all
        # objects)
        hmsg.save_masked_pcds(path=save_dir, state="both")
        hmsg.save_full_pcd(path=save_dir)
        hmsg.save_full_pcd_feats(path=save_dir)

        # # # # for debugging: load preconstructed map as follows
        # hmsg.load_full_pcd(path=save_dir)
        # hmsg.load_full_pcd_feats(path=save_dir)
        # hmsg.load_masked_pcds_new(path=save_dir)
        # create graph, only if dataset is not Replia or ScanNet
        print(params.main.dataset)
        hmsg.build_hier_multimodal_scene_graph(save_path=save_dir)

        params.main.dataset_path = dataset_path
        params.main.save_path = save_path


if __name__ == "__main__":
    main()
