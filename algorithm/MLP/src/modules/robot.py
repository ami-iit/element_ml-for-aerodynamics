import numpy as np
from idyntree import bindings as idyntree
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import toml


class Robot:
    def __init__(self, robot_name: str, urdf_path: str):
        self.name = robot_name
        self.urdf_path = urdf_path
        config_path = Path(__file__).parents[2] / "config" / f"{self.name}.toml"
        self.config = toml.load(config_path)
        self._load_configuration()
        self.kindyn = self._load_kindyn()

    def _load_configuration(self):
        self.joint_list = self.config["Joints"]["List"]
        self.base_link = self.config["Model"]["Base_link"]
        self.surface_list = self.config["Surfaces"]["List"]
        self.surface_frames = self.config["Surfaces"]["Frames"]
        self.surface_axes = self.config["Surfaces"]["Axes"]
        self.rotation_angles = self.config["Surfaces"]["Rot"]
        self.image_resolution = np.array(self.config["Surfaces"]["Resolutions"])

    def _load_kindyn(self):
        print(f"[loadReducedModel]: loading the following model: {self.urdf_path}")
        model_loader = idyntree.ModelLoader()
        reduced_model = model_loader.model()
        model_loader.loadReducedModelFromFile(str(self.urdf_path), self.joint_list)
        self.nDOF = reduced_model.getNrOfDOFs()
        kindyn = idyntree.KinDynComputations()
        kindyn.loadRobotModel(reduced_model)
        kindyn.setFloatingBase(self.base_link)
        print(
            f"[loadReducedModel]: loaded model: {self.urdf_path}, number of joints: {self.nDOF}"
        )
        return kindyn

    def set_state(self, pitch_angle, yaw_angle, joint_positions):
        # Compute base pose
        world_R_base = R.from_euler(
            "zxy", [-90, -pitch_angle, yaw_angle], degrees=True
        ).as_matrix()
        base_pose = np.block(
            [[world_R_base, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]
        )
        # Set unused variables
        joint_velocities = np.zeros_like(joint_positions)
        base_velocity = np.zeros(6)
        gravity_acceleration = np.array([0, 0, 9.81])
        # Set robot state
        ack1 = self.kindyn.setRobotState(
            base_pose,
            joint_positions,
            base_velocity,
            joint_velocities,
            gravity_acceleration,
        )
        # Check if the robot state is set correctly #2
        joint_positions_iDynTree = idyntree.VectorDynSize(len(joint_positions))
        self.kindyn.getJointPos(joint_positions_iDynTree)
        val = np.linalg.norm(joint_positions - joint_positions_iDynTree.toNumPy())
        ack2 = val < 1e-6
        if not ack1 and not ack2:
            print("[setRobotState]: error in setting robot state.")
        return

    def visualize(self):
        viz = idyntree.Visualizer()
        if viz.addModel(self.kindyn.model(), self.name):
            print("[initializeVisualizer]: model loaded in the visualizer.")
        else:
            print("[initializeVisualizer]: unable to load the model in the visualizer.")
        viz.camera().animator().enableMouseControl(True)
        base_pose = (
            self.kindyn.getWorldBaseTransform().asHomogeneousTransform().toNumPy()
        )
        joint_positions = idyntree.VectorDynSize(self.nDOF)
        self.kindyn.getJointPos(joint_positions)
        viz.modelViz(self.name).setPositions(base_pose, joint_positions)
        while viz.run():
            viz.draw()
        return

    def compute_world_H_link(self, frame_name):
        world_H_link = (
            self.kindyn.getWorldTransform(frame_name).asHomogeneousTransform().toNumPy()
        )
        return world_H_link

    def compute_all_world_H_link(self):
        world_H_link_dict = {}
        for surface_id, surface_name in enumerate(self.surface_list):
            world_H_link = self.compute_world_H_link(
                frame_name=self.surface_frames[surface_id]
            )
            frame_rotation_matrix = R.from_euler(
                "z", self.rotation_angles[surface_id], degrees=True
            ).as_matrix()
            world_H_link[:3, :3] = np.dot(world_H_link[0:3, 0:3], frame_rotation_matrix)
            world_H_link_dict[surface_name] = world_H_link
        return world_H_link_dict

    def invert_homogeneous_transform(self, a_H_b):
        a_R_b = a_H_b[0:3, 0:3]
        a_d_b = a_H_b[0:3, -1]
        b_H_a = np.block(
            [
                [a_R_b.T, np.dot(-a_R_b.T, a_d_b).reshape((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        )
        return b_H_a

    def compute_link_H_world(self, frame_name):
        world_H_link = self.compute_world_H_link(frame_name)
        link_H_world = self.invert_homogeneous_transform(world_H_link)
        return link_H_world

    def compute_all_link_H_world(self):
        link_H_world_dict = {}
        for surface_id, surface_name in enumerate(self.surface_list):
            world_H_link = self.compute_world_H_link(
                frame_name=self.surface_frames[surface_id]
            )
            frame_rotation_matrix = R.from_euler(
                "z", self.rotation_angles[surface_id], degrees=True
            ).as_matrix()
            world_H_link[:3, :3] = np.dot(world_H_link[0:3, 0:3], frame_rotation_matrix)
            link_H_world = self.invert_homogeneous_transform(world_H_link)
            link_H_world_dict[surface_name] = link_H_world
        return link_H_world_dict

    def load_mesh(self):
        meshes = []
        model = self.kindyn.getRobotModel()
        visuals = model.visualSolidShapes().getLinkSolidShapes()
        for link_id in range(model.getNrOfLinks()):
            link_name = model.getLinkName(link_id)
            for link_visual in visuals[link_id]:
                if link_visual.isExternalMesh():
                    mesh_path = (
                        link_visual.asExternalMesh().getFileLocationOnLocalFileSystem()
                    )
                    mesh_name = f"{link_visual.asExternalMesh().getFilename()}"
                    link_H_mesh = (
                        link_visual.getLink_H_geometry()
                        .asHomogeneousTransform()
                        .toNumPy()
                    )
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    mesh.scale(0.001, center=[0, 0, 0])  # From [m] to [mm]
                    world_H_link = self.compute_world_H_link(frame_name=link_name)
                    world_H_mesh = np.dot(world_H_link, link_H_mesh)
                    mesh.transform(world_H_mesh)
                    mesh.compute_vertex_normals()
                    meshes.append({"name": mesh_name, "mesh": mesh})
        return meshes
