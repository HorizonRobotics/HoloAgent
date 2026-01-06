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
from copy import deepcopy

from memory.hmsg.graph.graph import Graph
import hydra
import open3d as o3d
from omegaconf import DictConfig
import time
import numpy as np
import os
import json
# pylint: disable=all


def visualize_and_save(room_pcd, obj_pcd, end_sphere, save_path="scene.png"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)  # 设置 False 可后台渲染
    vis.add_geometry(room_pcd)
    vis.add_geometry(obj_pcd)
    vis.add_geometry(end_sphere)

    vis.poll_events()
    vis.update_renderer()

    # 计算房间和物体的中心
    room_center = np.array(room_pcd.get_center())
    obj_center = np.array(obj_pcd.get_center())

    # 相机位置：在房间中心 + y方向上方 5m
    cam_pos = room_center + np.array([0, 5.0, 0.0])

    # 设置相机参数
    ctr = vis.get_view_control()
    ctr.set_lookat(obj_center)                     # 看向物体中心
    ctr.set_front((cam_pos - obj_center) /
                  np.linalg.norm(cam_pos - obj_center))  # 相机朝向
    ctr.set_up([0, 1, 0])                          # 这里假设 z 作为水平参考，上方向定为 z

    ctr.set_zoom(0.7)  # 缩放调节

    vis.poll_events()
    vis.update_renderer()

    # 保存截图
    vis.capture_screen_image(save_path)
    vis.destroy_window()
    print(f"Saved visualization to {save_path}")


instruction_templelate_sh3f_obj = [
    # office-13
    "Can you find me a chair",
    "Can you find me a sofa",
    "Can you find me a desk",
    "Can you fine me a computer",
    "Find me a banana",
    "I want to eat some fruit.",
    "Find me a cup",
    "Find me a white table",
    "Find me a white chair",
    "Find me a mechine arm robot",
    "Find me a trash can",
    "Find me a laptop",
    "Find me a book",

    # # small obj - 4
    "I am thirsty, can you find me something to drink",  # demand type
    "Find me Pepsi-Cola",
    "Find me Coca-Cola",
    "Find me a can",

    # pantry - 2
    "Take me to the coffee machine",
    "Take me to the cabinet",

    # office pantry -2
    "Take me to the table",
    "Take me to the chair",
    # small obj -2
    "Fine me some animal toy",
    "Take me to the kettle",
]

instruction_templelate_sh3f = [  # 23
    # office
    "Can you find me a chair in the office",
    "Can you find me a sofa in the office",
    "Can you find me a desk in the office",
    "Can you fine me a computer in the office",
    "Find me a banana in the office",
    "I want to eat some fruit.",
    "Find me a cup in the office",
    "Find me a white table in the office",
    "Find me a white chair",
    "Find me a mechine arm robot in on the desk in the office",
    "Find me a trash can in the office",
    "Find me a laptop in the office",
    "Find me a book in the office",

    # # small obj
    "I am thirsty, can you find me something to drink in the office",  # demand type
    "Find me Pepsi-Cola in the office",
    "Find me Coca-Cola in the office",
    "Find me a can the office",

    # # pantry
    "Take me to the coffee machine in the office pantry",
    "Take me to the cabinet in the office pantry",

    # # office pantry
    "Take me to the table in the office pantry",
    "Take me to the chair in the office pantry",
    # small obj
    "Fine me some animal toy in the office pantry",
    "Take me to the kettle in the office pantry",
]


instruction_templelate_sh3f_autoregion = [  # 23
    # office
    "Can you find me a chair in the office",
    "Can you find me a sofa in the office",
    "Can you find me a desk in the office",
    "Can you fine me a computer in the office",
    "Find me a banana in the office",
    "I want to eat some fruit.",
    "Find me a cup in the office",
    "Find me a white table in the office",
    "Find me a white chair",
    "Find me a mechine arm robot in on the desk in the office",
    "Find me a trash can in the office",
    "Find me a laptop in the office",
    "Find me a book in the office",

    # # small obj
    "I am thirsty, can you find me something to drink in the office",  # demand type
    "Find me Pepsi-Cola in the office",
    "Find me Coca-Cola in the office",
    "Find me a can the office",

    # # pantry
    "Take me to the coffee machine in the office pantry",
    "Take me to the cabinet in the office pantry",

    # # office pantry
    "Take me to the table in the office pantry",
    "Take me to the chair in the office pantry",
    # small obj
    "Fine me some animal toy in the office pantry",
    "Take me to the kettle in the office pantry",
]

final_instruction_telepalte = instruction_templelate_sh3f_autoregion


@hydra.main(version_base=None, config_path="../../config",
            config_name="visualize_query_graph_icra_sh3f")  # label, obj-embedding, view-embedding
def main(params: DictConfig):
    # Load graph
    scene_id = params.main.scene_id
    use_gpt = params.main.use_gpt
    # "autoregion"  # "none" # "human_assign" #
    spatial_reasoning_method = params.main.spatial_reasoning_method
    fast_slow_method = params.main.fast_slow_method
    if fast_slow_method == "fast_match":
        use_gpt = False
    # Create save directory
    params.main.dataset_path = os.path.join(
        params.main.dataset_path,
        scene_id)  # params.main.scene_id
    save_dir = os.path.join(
        params.main.save_path,
        params.main.dataset,
        scene_id)  # params.main.scene_id
    params.main.save_path = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print("dataset_path: ", params.main.dataset_path)
    print("save_path: ", save_dir)
    hmsg = Graph(params)
    hmsg.load_hmsg_graph(params.main.graph_path)
    hmsg.vln_result_dir = os.path.join(
        save_dir, f"fsrvln_result_online_{spatial_reasoning_method}_{fast_slow_method}")
    # 自主判断房间类型和名字
    hmsg.generate_room_names(
        # generate_method="view_embedding",
        # generate_method="label",
        generate_method="obj_embedding",
        default_room_types=[
            "Pantry",
            "Office",
            "Office-Pantry",
        ]
    )
    # 人为设定房间类型和名字
    if spatial_reasoning_method == "human_assign":
        designated_room_names_sh3f = [
            "office pantry",
            "office pantry",
            "office",
        ]
        hmsg.set_room_names(room_names=designated_room_names_sh3f)
        final_instruction_telepalte = instruction_templelate_sh3f
    elif spatial_reasoning_method == "auto_region":
        final_instruction_telepalte = instruction_templelate_sh3f_autoregion
    else:
        final_instruction_telepalte = instruction_templelate_sh3f_obj

    T_switch_axis = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [
                             0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)  # map to hmsg
    T_tomap = np.linalg.inv(T_switch_axis)  # hmsg to map
    # print("T_tomap: ", T_tomap)
    # loop forever and ask for query, until user click 'q'
    json_save_path = os.path.join(hmsg.vln_result_dir, "all_results.json")
    all_results = []  # 存放每条 query 的结果

    sum_Total_Time = 0.0
    sum_FastMatching = 0.0
    sum_ObjectInImageCheck = 0.0
    sum_VLM_Rethinking = 0.0
    sum_Re_Matching = 0.0
    sum_LLM_parse = 0.0

    for query_instruction in final_instruction_telepalte:
        print(query_instruction)
        hmsg.curr_query_save_dir = os.path.join(
            hmsg.vln_result_dir, query_instruction)
        if not os.path.exists(hmsg.curr_query_save_dir):
            os.makedirs(hmsg.curr_query_save_dir)

        start_time = time.time()

        floor, room, obj, res_dict = hmsg.query_hierarchy_protected_icra(
            query_instruction, top_k=5, use_gpt=use_gpt)
        end_time = time.time()
        query_time = end_time - start_time
        print(f"运行时间: {query_time:.4f} 秒")
        # visualize the query
        print(floor.floor_id, [(r.room_id, r.name)
              for r in room], [o.object_id for o in obj])
        # 构建要写入 JSON 的数据
        query_result = {
            "query": query_instruction,
            "time_seconds": query_time,
            "floor_id": floor.floor_id,
            "rooms": [{"room_id": r.room_id, "name": r.name} for r in room],
            "objects": [{"object_id": o.object_id} for o in obj]
        }
        # use open3d to visualize room.pcd and color the points where obj.pcd
        # is
        print("len(obj): ", len(obj))
        for i in range(len(obj)):
            obj_pcd = obj[i].pcd.paint_uniform_color([0, 1, 0])  # rgb
            room_pcd = room[i].pcd
            obj_pcd = deepcopy(obj[i].pcd)
            room_pcd = deepcopy(room[i].pcd)
            obj_center = obj_pcd.get_center()
            print("obj_center in scenegraph: ", obj_center)
            obj_center_h = np.hstack((obj_center, 1.0))  # 齐次坐标 (4,)
            obj_center_in_map = (T_tomap @ obj_center_h)[:3]
            print("obj_center in lidarmap: ", obj_center_in_map)

            end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            end_sphere.translate(obj_center)
            end_sphere.paint_uniform_color([1, 0, 0])
            # o3d.visualization.draw_geometries([room_pcd, obj_pcd, end_sphere])
            # 合并点云
            mesh_pcd = end_sphere.sample_points_uniformly(number_of_points=500)
            combined_pcd = room_pcd + obj_pcd + mesh_pcd
            # 保存为单个文件
            pcd_save_path = os.path.join(
                hmsg.curr_query_save_dir, f"scene_{i}.ply")
            pcd_render_save_path = os.path.join(
                hmsg.curr_query_save_dir, f"scene_{i}.png")
            o3d.io.write_point_cloud(pcd_save_path, combined_pcd)
            visualize_and_save(
                room_pcd,
                obj_pcd,
                end_sphere,
                save_path=pcd_render_save_path)
            print(f"Saved {pcd_save_path}")
        all_results.append(query_result)

        # res_dict["FastMatching"] = FastMatching_time
        # res_dict["ObjectInImageCheck"] = 0.0
        # res_dict["VLM_Rethinking"] = 0.0
        # res_dict["Re_Matching"] = 0.0
        # res_dict["Total_Time"] = total_online_query_time
        sum_LLM_parse = sum_LLM_parse + res_dict["LLM_Parse_Time"]
        sum_Total_Time = sum_Total_Time + res_dict["Total_Time"]
        sum_FastMatching = sum_FastMatching + res_dict["FastMatching"]
        sum_ObjectInImageCheck = sum_ObjectInImageCheck + \
            res_dict["ObjectInImageCheck"]
        sum_VLM_Rethinking = sum_VLM_Rethinking + res_dict["VLM_Rethinking"]
        sum_Re_Matching = sum_Re_Matching + res_dict["Re_Matching"]

    average_fastmatching_time = sum_FastMatching / \
        len(final_instruction_telepalte)
    average_objectinimagecheck_time = sum_ObjectInImageCheck / \
        len(final_instruction_telepalte)
    average_vlm_rethinking_time = sum_VLM_Rethinking / \
        len(final_instruction_telepalte)
    average_re_matching_time = sum_Re_Matching / \
        len(final_instruction_telepalte)
    average_total_time = sum_Total_Time / len(final_instruction_telepalte)
    average_llm_parse_time = sum_LLM_parse / len(final_instruction_telepalte)

    print(f"fsrvln average_total_time : {average_total_time:.4f} 秒")
    print(
        f"fsrvln average_objectinimagecheck_time : {average_objectinimagecheck_time:.4f} 秒")
    print(
        f"fsrvln average_vlm_rethinking_time : {average_vlm_rethinking_time:.4f} 秒")
    print(
        f"fsrvln average_re_matching_time : {average_re_matching_time:.4f} 秒")
    print(
        f"fsrvln average_fastmatching_time : {average_fastmatching_time:.4f} 秒")
    print(f"fsrvln average_llm_parse_time : {average_llm_parse_time:.4f} 秒")
    # 将平均时间也写入 JSON
    final_json = {
        "average_total_time": average_total_time,
        "average_objectIncheck_time": average_objectinimagecheck_time,
        "average_vlm_rethinking_time": average_vlm_rethinking_time,
        "average_re_matching_time": average_re_matching_time,
        "average_fastmatching_time": average_fastmatching_time,
        "average_llm_parse_time": average_llm_parse_time,
        "results": all_results
    }
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print(f"All results saved to {json_save_path}")


if __name__ == "__main__":
    main()
