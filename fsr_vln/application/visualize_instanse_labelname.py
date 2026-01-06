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
# pylint: disable=missing-docstring
import time
# 读取点云
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# 用于创建图形用户界面和渲染场景
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore


def exit_after_delay():
    time.sleep(5)  # 等待 5 秒
    gui.Application.instance.quit()  # 退出应用

# 函数功能，用于显示点云并标注类别标签


def show_point_cloud_with_labels(pcd, point_labels, cluster_labels):
    """
    显示点云并标注类别标签。

    Args:
        pcd (open3d.geometry.PointCloud): 输入的点云数据。
        labels (numpy.ndarray): 点云的聚类标签，每个点对应一个聚类索引。
        cluster_names (dict): 聚类索引到类名的映射，用于显示在点云标签中。

    Returns:
        None
    """
    # 初始化GUI应用
    app = gui.Application.instance
    app.initialize()

    # 创建窗口和场景
    window = app.create_window(
        "mapvln raw existing object instances", 1024, 768)
    # 创建一个场景小部件（SceneWidget）并将其添加到窗口中
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # 设置场景背景和光照
    scene.scene.set_background([1, 1, 1, 1])  # 白色背景
    scene.scene.add_geometry("pcd", pcd, rendering.MaterialRecord())  # 添加点云到场景

    # 遍历每个聚类标签，并在点云中添加对应的文本标签
    for i in range(max(point_labels) + 1):
        cluster_idx = np.where(point_labels == i)[0]  # 找到术语当前聚类的索引
        if len(cluster_idx) == 0:
            continue
        cluster_points = np.asarray(pcd.points)[cluster_idx]
        center = cluster_points.mean(axis=0)  # 计算当前家具类点的中心位置
        # 获取聚类名称，如果没有则使用默认名称cluster_{i}
        label = cluster_labels.get(i, f"cluster_{i}")
        # 添加3d文本标签
        scene.add_3d_label(center, label)

    # 设置相机的中心位置为[0, 0, 0]
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1)
    # 获取点云的轴对齐包围盒
    bounding_box = pcd.get_axis_aligned_bounding_box()
    # 设置相机视角为60度，并将其对准点云的包围盒
    scene.setup_camera(60.0, bounding_box, center)

    # 启动GUI应用，显示窗口
    app.run()


# 加载点云和聚类
pcd = o3d.io.read_point_cloud(
    "/mnt/disk2/hovsg/HOV-SG/data/scannet/scene_graph/scannet/scene0378_00/full_pcd.ply")
labels = np.array(pcd.cluster_dbscan(eps=0.05,  # 聚类半径
                                     min_points=50,  # 最小点数
                                     print_progress=True))  # 显示进度

# 颜色
max_label = labels.max()
print("max_label: ", max_label)
colors = plt.get_cmap("tab20")(
    labels / (max_label + 1 if max_label > 0 else 1))
print(colors)
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 聚类类别名映射
cluster_names = {0: "chair", 1: "table", 2: "sofa"}  # 可根据聚类结果自行扩展

# 显示
# 启动一个线程，在 5 秒后退出应用
# threading.Thread(target=exit_after_delay).start()
show_point_cloud_with_labels(pcd, labels, cluster_names)  # 调用时保持变量名一致
