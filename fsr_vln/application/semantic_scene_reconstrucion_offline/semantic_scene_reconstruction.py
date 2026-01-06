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

# pylint: disable=E,W,R,F
"""
semantic_scene_reconstruction_unified.

基于传入的配置对不同场景执行语义重建并构建多模态场景图（HMSG）。
脚本通过 Hydra 加载配置，创建 `Graph` 实例，
生成并保存特征地图、点云、掩码点云与特征文件，然后调用图构建流程
将结果写入磁盘。

注意
----
- 依赖项目中的 `hmsg.graph.graph.Graph` 实现与配置文件 `config/`
    下的相应配置文件。
- 运行时会在磁盘上创建并写入输出目录（具有副作用）。
- 可以通过命令行参数指定不同的配置文件，例如：
  python semantic_scene_reconstruction_unified.py --config-name=semantic_scene_reconstruction_ic3f
"""


# 添加项目根目录到Python路径
from memory.hmsg.graph.graph import Graph
import os
import hydra
from omegaconf import DictConfig
import sys
sys.path.insert(
    0, os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))))


def run_scene_reconstruction(params: DictConfig):
    """
    执行单个场景的语义重建与场景图构建的核心函数。

    根据 Hydra 注入的 `params` 初始化并运行 `Graph` 的重建流程。主要步骤：
    1. 基于 `params.main.scene_id` 与 `params.main.dataset_path` 计算输入/输出路径；
    2. 创建输出目录并初始化 `Graph(params)`；
    3. 生成特征地图并保存点云、掩码点云与特征；
    4. 调用 `build_hier_multimodal_scene_graph` 构建并保存多模态场景图。

    Args:
        params (DictConfig): Hydra 配置对象，期望包含 `params.main.scene_id`、
            `params.main.dataset_path`、`params.main.save_path` 和 `params.main.dataset` 等字段。

    Returns:
        None

    Side effects:
        在磁盘上创建并写入 `save_dir`，包含点云、特征以及生成的 graph 数据。
    """
    scene_ids = [params.main.scene_id]  # 使用配置文件中指定的场景ID

    if hasattr(
            params,
            'main') and hasattr(
            params.main,
            'scene_ids') and params.main.scene_ids:
        # 如果配置文件中定义了多个场景ID，则使用它们
        scene_ids = params.main.scene_ids

    for scene_id in scene_ids:
        # 更新参数中的scene_id
        if hasattr(params, 'main'):
            params.main.scene_id = scene_id
        # Create save directory
        params.main.dataset_path = os.path.join(
            params.main.dataset_path, scene_id)
        save_dir = os.path.join(
            params.main.save_path,
            params.main.dataset,
            scene_id)
        params.main.save_path = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        print("dataset_path: ", params.main.dataset_path)
        print("save_path: ", save_dir)
        # Create graph
        hmsg = Graph(params)

        # Semantic scene reconstruction
        hmsg.create_feature_map()

        # Save full point cloud, features, and masked point clouds (pcd for all
        # objects)
        hmsg.save_masked_pcds(path=save_dir, state="both")
        hmsg.save_full_pcd(path=save_dir)
        hmsg.save_full_pcd_feats(path=save_dir)

        # # # # For debugging: load preconstructed map as follows
        # hmsg.load_full_pcd(path=save_dir)
        # hmsg.load_full_pcd_feats(path=save_dir)
        # hmsg.load_masked_pcds_new(path=save_dir)

        # Create hierarchical multimodal scene graph
        print(params.main.dataset)
        hmsg.build_hier_multimodal_scene_graph(save_path=save_dir)


@hydra.main(version_base=None, config_path="../../config",
            config_name="semantic_scene_reconstruction_ic4f")  # 默认使用ic4f配置
def main(params: DictConfig):
    """
    主函数，通过 Hydra 加载配置并执行场景重建。

    可以通过命令行参数指定不同的配置文件，例如： python semantic_scene_reconstruction.py
    --config-name=semantic_scene_reconstruction_sh3f
    """
    run_scene_reconstruction(params)


if __name__ == "__main__":
    main()
