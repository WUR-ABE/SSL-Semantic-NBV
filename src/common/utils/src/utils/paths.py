"""Central location for file-system paths used across the workspace.

Configure via environment variables instead of editing source files:

    NBV_DATA_ROOT   Root of the dataset (simulation, real world, checkpoints).
                    Default: ~/nbv_data
    NBV_MODELS      Directory holding trained checkpoints, one sub-folder per
                    <experiment>_<target>_<planner> run.
                    Default: $NBV_DATA_ROOT/checkpoints
    NBV_MATERIALS   Directory holding evaluation results.
                    Default: <drl_nbv_plan package>/materials
    SAM3_CHECKPOINT Path to the SAM3 model weights (only for the SAM3 pipeline).
                    Default: <sam3_detector package>/materials/sam3.pt

Layout of $NBV_DATA_ROOT:

    simulation/                            simulated plants 1-20
        meshes/plant<i>.ply                    mesh exported from GroIMP (input of step 1)
        meshes/plant<i>.dae                    same mesh converted for Gazebo (input of step 2)
        gt_nodes/plant<i>_export_{fruit,internode}.csv   fruit/node annotations
        gt_pcd/plant<i>.pcd                    ground-truth point clouds (created by step 1)
        views/plant<i>/xyzr_<x>_<y>_<z>_<r>/{pcd,view}/   rendered views (created by step 2)
    real_world/plant<i>/                   everything of one real plant (31-40):
        viewpoints.npy                         camera poses of the recorded views
        pcd/pcd_<h>_<w>.npy                    recorded point cloud per view
        pcd_sam3/pcd_<h>_<w>.npy               same, with the SAM3 fruit label per point
        gt_pcd.pcd                             ground-truth point cloud
        gt_fruit.csv, gt_internode.csv         fruit/node annotations
        raw/                                   original recordings, not in the download:
            views/ (images, point clouds, poses), rgb/ and masks/ (shown in RViz,
            input and output of the SAM3 detector), gt_scan/ (original ground-truth scan)
    checkpoints/<experiment>_<target>_<planner>/   trained models
"""

import os
import time


def pkg_path(pkg_name):
    """Absolute path of a ROS package in this workspace."""
    try:
        import rospkg

        return rospkg.RosPack().get_path(pkg_name)
    except Exception:
        # Fallback when ROS is not sourced: this file lives in
        # <ws>/src/common/utils/src/utils/paths.py
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        for root, dirs, files in os.walk(src_dir):
            if os.path.basename(root) == pkg_name and "package.xml" in files:
                return root
        raise


DATA_ROOT = os.path.expanduser(os.environ.get("NBV_DATA_ROOT", "~/nbv_data"))
MATERIALS_DIR = os.path.expanduser(
    os.environ.get("NBV_MATERIALS", os.path.join(pkg_path("drl_nbv_plan"), "materials"))
)
MODELS_DIR = os.path.expanduser(
    os.environ.get("NBV_MODELS", os.path.join(DATA_ROOT, "checkpoints"))
)

SIM_DATA_DIR = os.path.join(DATA_ROOT, "simulation")
MESH_DIR = os.path.join(SIM_DATA_DIR, "meshes")
GT_NODES_DIR = os.path.join(SIM_DATA_DIR, "gt_nodes")
GT_PCD_DIR = os.path.join(SIM_DATA_DIR, "gt_pcd")
# Per-view point clouds rendered in Gazebo (written by collect_view_data.py,
# read by the simulation environment).
VIEW_DATA_DIR = os.path.join(SIM_DATA_DIR, "views")
REAL_DATA_DIR = os.path.join(DATA_ROOT, "real_world")

REAL_PLANTS = list(range(31, 41))


def real_plant_dir(plant):
    """Folder of one real plant: $NBV_DATA_ROOT/real_world/plant<i>."""
    return os.path.join(REAL_DATA_DIR, "plant%s" % plant)


def gt_pcd_file(plant):
    """Ground-truth point cloud of a simulated (1-20) or real (31-40) plant."""
    if int(plant) in REAL_PLANTS:
        return os.path.join(real_plant_dir(plant), "gt_pcd.pcd")
    return os.path.join(GT_PCD_DIR, "plant%s.pcd" % plant)


def gt_nodes_file(plant, kind):
    """Annotation file of a plant; kind is 'fruit' or 'internode'."""
    if int(plant) in REAL_PLANTS:
        return os.path.join(real_plant_dir(plant), "gt_%s.csv" % kind)
    return os.path.join(GT_NODES_DIR, "plant%s_export_%s.csv" % (plant, kind))


def model_dir(experiment, recon_target, planner, models_dir=None):
    """Folder of one training run: <models_dir>/<experiment>_<target>_<planner>."""
    return os.path.join(models_dir or MODELS_DIR, "%s_%s_%s" % (experiment, recon_target, planner))


def results_dir(name):
    """Fresh, timestamped output folder: $NBV_MATERIALS/results/<name>/<timestamp>."""
    return os.path.join(MATERIALS_DIR, "results", name, time.strftime("%Y%m%d_%H%M%S"))
