import os
import torch
import numpy as np


class FlaviaDatasetSDP:
    def __init__(self, dir_path, resize=None, aug_prob=0.0):
        self.aug_prob = aug_prob
        self.resize = resize
        self.files_path = [
            os.path.join(dir_path, img) for img in sorted(os.listdir(dir_path))
        ]
        self.labels = [
            label.split("_")[-1].split(".")[0]
            for label in sorted(os.listdir(dir_path))
        ]
        self.classes = [
            "pubescent bamboo",
            "Chinese horse chestnut",
            "Anhui Barberry",
            "Chinese redbud",
            "true indigo",
            "Japanese maple",
            "Nanmu",
            "castor aralia",
            "Chinese cinnamon",
            "goldenrain tree",
            "Big-fruited Holly",
            "Japanese cheesewood",
            "wintersweet",
            "camphortree",
            "Japan Arrowwood",
            "sweet osmanthus",
            "deodar",
            "maidenhair tree",
            "Crape myrtle Crepe myrtle",
            "oleander",
            "yew plum pine",
            "Japanese Flowering Cherry",
            "Glossy Privet",
            "Chinese Toon",
            "peach",
            "Ford Woodlotus",
            "trident maple",
            "Beale's barberry",
            "southern magnolia",
            "Canadian poplar",
            "Chinese tulip tree",
            "tangerine",
        ]

    def random_noise(self, control_points, noise_factor=0.02):
        noise = np.random.normal(0, noise_factor, control_points.shape)
        return control_points + noise

    def random_rotation(self, control_points):
        angle = np.random.uniform(
            -45, 45
        )  # You can adjust the range of rotation
        rotation_matrix = np.array(
            [
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))],
            ]
        )
        return np.dot(control_points, rotation_matrix)

    def random_flip(self, control_points, flip_type="horizontal"):
        if flip_type == "horizontal":
            return np.array([[-p[0], p[1]] for p in control_points])
        elif flip_type == "vertical":
            return np.array([[p[0], -p[1]] for p in control_points])

    def __len__(self):
        return len(self.files_path)

    def normalize(self, control_points):
        mean = np.mean(control_points, axis=0)
        std = np.std(control_points, axis=0)
        return (control_points - mean) / std

    def __getitem__(self, idx):
        with open(self.files_path[idx], "r") as file:
            lines = [
                line.strip()
                for line in file.readlines()
                if line.strip() and not line.startswith("#")
            ]
            # Assuming each non-commented line represents a point with x and y coordinates
            try:
                points = [tuple(map(float, line.split())) for line in lines]
            except:
                points = [eval(line) for line in lines]
            x, y = zip(*points)
            spline_pts = np.column_stack((x, y))

        label = int(self.labels[idx])
        if np.random.rand() < self.aug_prob:
            # spline_pts = self.random_noise(spline_pts, 2)
            # spline_pts = self.random_flip(spline_pts)
            spline_pts = self.random_rotation(spline_pts)
        return {
            "input": torch.tensor(spline_pts, dtype=torch.float),
            "target": torch.tensor(label, dtype=torch.long),
        }


class FashionMNISTSDP:
    def __init__(self, dir_path, resize=None, aug_prob=0.0):
        self.resize = resize
        self.aug_prob = aug_prob
        self.files_path = [
            os.path.join(dir_path, i) for i in sorted(os.listdir(dir_path))
        ]
        self.labels = [
            i.split("_")[-1].split(".")[0]
            for i in sorted(os.listdir(dir_path))
        ]
        self.classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

    def __len__(self):
        return len(self.files_path)

    def random_noise(self, control_points, noise_factor=0.01):
        noise = np.random.normal(0, noise_factor, control_points.shape)
        return control_points + noise

    def random_rotation(self, control_points):
        angle = np.random.uniform(
            -30, 30
        )  # You can adjust the range of rotation
        rotation_matrix = np.array(
            [
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))],
            ]
        )
        return np.dot(control_points, rotation_matrix)

    def random_flip(self, control_points, flip_type="horizontal"):
        if flip_type == "horizontal":
            return np.array([[-p[0], p[1]] for p in control_points])
        elif flip_type == "vertical":
            return np.array([[p[0], -p[1]] for p in control_points])

    def __getitem__(self, idx):
        with open(self.files_path[idx], "r") as file:
            lines = [
                line.strip()
                for line in file.readlines()
                if line.strip() and not line.startswith("#")
            ]
            try:
                points = [tuple(map(float, line.split())) for line in lines]
            except:
                points = [eval(line) for line in lines]
            x, y = zip(*points)
            spline_pts = np.column_stack((x, y))
        label = int(self.labels[idx])
        if np.random.rand() < self.aug_prob:
            spline_pts = self.random_noise(spline_pts, 2)
            # spline_pts = self.random_rotation(spline_pts)
        return {
            "input": torch.tensor(spline_pts, dtype=torch.float),
            "target": torch.tensor(label, dtype=torch.long),
        }


class FolioSDP:
    def __init__(self, dir_path, resize=None, aug_prob=0.0):
        self.root_dir = dir_path
        self.resize = resize
        self.aug_prob = aug_prob
        self.classes = sorted(
            cls_name
            for cls_name in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, cls_name))
        )
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }
        self.samples = self._make_dataset()

    def _make_dataset(self):
        return [
            (
                os.path.join(self.root_dir, class_name, file_name),
                self.class_to_idx[class_name],
                class_name,
            )
            for class_name in self.classes
            for file_name in os.listdir(
                os.path.join(self.root_dir, class_name)
            )
        ]

    def __len__(self):
        return len(self.samples)

    def random_noise(self, control_points, noise_factor=0.01):
        noise = np.random.normal(0, noise_factor, control_points.shape)
        return control_points + noise

    def random_rotation(self, control_points):
        angle = np.random.uniform(
            -30, 30
        )  # You can adjust the range of rotation
        rotation_matrix = np.array(
            [
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))],
            ]
        )
        return np.dot(control_points, rotation_matrix)

    def random_flip(self, control_points, flip_type="horizontal"):
        if flip_type == "horizontal":
            return np.array([[-p[0], p[1]] for p in control_points])
        elif flip_type == "vertical":
            return np.array([[p[0], -p[1]] for p in control_points])

    def __getitem__(self, idx):
        files_path, label, class_name = self.samples[idx]
        with open(files_path, "r") as file:
            lines = [
                line.strip()
                for line in file.readlines()
                if line.strip() and not line.startswith("#")
            ]
            try:
                points = [tuple(map(float, line.split())) for line in lines]
            except:
                points = [eval(line) for line in lines]
            x, y = zip(*points)
            spline_pts = np.column_stack((x, y))
        if np.random.rand() < self.aug_prob:
            # spline_pts = self.random_noise(spline_pts, 2)
            spline_pts = self.random_rotation(spline_pts)
        return {
            "input": torch.tensor(spline_pts, dtype=torch.float),
            "target": torch.tensor(label, dtype=torch.long),
        }


