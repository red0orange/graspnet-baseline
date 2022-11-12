import os
import sys
import time
import random
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

o3d.visualization.webrtc_server.enable_webrtc()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "dataset"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

import rospy
import tf
from tf import transformations as T
import message_filters
from cv_bridge import CvBridge
import sensor_msgs.msg
from std_msgs.msg import String


def get_net(cfgs):
    # Init the model
    net = GraspNet(
        input_feature_dim=0,
        num_view=cfgs.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"]
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(
        gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh
    )
    gg = gg[~collision_mask]
    return gg


def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    # o3d.visualization.draw_geometries([cloud, *grippers])
    o3d.visualization.draw([cloud, *grippers])
    pass


class GraspNet1BillionInterface:
    def __init__(self, rgb_topic_name, depth_topic_name, camera_info_topic_name, net):
        rospy.init_node("graspnet_1billion_interface")

        # infer
        self.net = net

        # Start ROS subscriber
        self.cv_bridge = CvBridge()
        self.tf_br = tf.TransformBroadcaster()
        image_sub = message_filters.Subscriber(rgb_topic_name, sensor_msgs.msg.Image)
        depth_sub = message_filters.Subscriber(depth_topic_name, sensor_msgs.msg.Image)
        info_sub = message_filters.Subscriber(camera_info_topic_name, sensor_msgs.msg.CameraInfo)
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub, info_sub], 1, slop=1
        )
        ts.registerCallback(self.image_callback)
        pass

    def image_test(self, image, depth, camera_info):
        gg = self.infer(image, depth, camera_info)
        rotation_matrices = gg.rotation_matrices
        translations = gg.translations
        num = translations.shape[0]

        while True:
            time.sleep(1)
            for i in range(num):
                T_matrice = np.identity(4)
                T_matrice[:3, :3] = rotation_matrices[i]
                T_matrice[:3, -1] = translations[i]

                trans = T.translation_from_matrix(T_matrice)
                quaternion = T.quaternion_from_matrix(T_matrice)
                self.tf_br.sendTransform(
                    trans,
                    quaternion,
                    rospy.Time.now(),
                    "grasp_pose_{}".format(i),
                    "camera_color_optical_frame",
                )
                pass
            pass

    def image_callback(self, image_msg, depth_msg, camera_info):
        """Image callback"""
        rgb_image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        K = np.array(camera_info.K).reshape([3, 3])
        image_width, image_height = rgb_image.shape[:2][::-1]
        camera_info = {
            "fx": K[0, 0],
            "fy": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
            "image_width": image_width,
            "image_height": image_height,
            "factor_depth": 1000.0,
        }

        if not self.flag:
            self.flag = True

            rgb_image = np.array(rgb_image, dtype=np.float32) / 255.0
            gg = self.infer(rgb_image, depth_image, camera_info)
            rotation_matrices = gg.rotation_matrices
            translations = gg.translations

            while True:
                time.sleep(1)
                for i, (rotation_matrice, translation) in enumerate(
                    rotation_matrices, translations
                ):
                    T_matrice = np.diag([4, 4], dtype=np.float32)
                    T_matrice[:3, :3] = rotation_matrice
                    T_matrice[:3, -1] = translation

                    trans = T.translation_from_matrix(T_matrice)
                    quaternion = T.quaternion_from_matrix(T_matrice)
                    self.tf_br.sendTransform(
                        trans,
                        quaternion,
                        rospy.Time.now(),
                        "grasp_pose_{}".format(i),
                        "camera_color_optical_frame",
                    )
                    pass
                pass

    def data_to_net_input(self, rgb, depth, camera_info):
        camera = CameraInfo(
            camera_info["image_width"],
            camera_info["image_height"],
            camera_info["fx"],
            camera_info["fy"],
            camera_info["cx"],
            camera_info["cy"],
            camera_info["factor_depth"],
        )
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (depth < 1000) & (depth > 0)
        cloud_masked = cloud[mask]
        color_masked = rgb[mask]

        # sample points
        if len(cloud_masked) >= cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(
                len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True
            )
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points["point_clouds"] = cloud_sampled
        end_points["cloud_colors"] = color_sampled

        return end_points, cloud

    def infer(self, rgb, depth, camera_info):
        end_points, cloud = self.data_to_net_input(rgb, depth, camera_info)
        gg = get_grasps(self.net, end_points)
        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))
        # downsample
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        return gg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default="/home/dehao/Projects/GraspFramework/graspnet-baseline/checkpoints/checkpoint-rs.tar",
        required=False,
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--num_point", type=int, default=20000, help="Point Number [default: 20000]"
    )
    parser.add_argument(
        "--num_view", type=int, default=300, help="View Number [default: 300]"
    )
    parser.add_argument(
        "--collision_thresh",
        type=float,
        default=0.01,
        help="Collision Threshold in collision detection [default: 0.01]",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel Size to process point clouds before collision detection [default: 0.01]",
    )
    cfgs = parser.parse_args()

    net = get_net(cfgs)
    interface = GraspNet1BillionInterface(
        "/camera/color/image_raw",
        "/camera/aligned_depth_to_color/image_raw",
        "/camera/aligned_depth_to_color/camera_info",
        net,
    )

    # image test
    rgb_path = (
        "/home/dehao/Projects/GraspFramework/graspnet-baseline/test_data/0_color.png"
    )
    depth_path = (
        "/home/dehao/Projects/GraspFramework/graspnet-baseline/test_data/0_depth.png"
    )
    color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_path))
    camera_info = {
        "fx": 908.5330810546875,
        "fy": 909.3223876953125,
        "cx": 647.282470703125,
        "cy": 354.7244873046875,
        "image_width": 1280,
        "image_height": 720,
        "factor_depth": 1000.0,
    }
    interface.image_test(color, depth, camera_info)

    # real ros demo
    rospy.spin()
    pass
