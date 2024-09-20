import os
import cv2 
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
        )  
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
    
    def filter_points(self, x, y, threshold=5):
        x = x[x > threshold]
        y = y[y > threshold] 
        return x, y 

    def generate_gt(self, points, idx):  
        x = points[:, 0]
        y = points[:, 1]
        x, y = self.filter_points(x, y, 10)
        img_height = int(max(y)) + 10
        img_width = int(max(x)) + 10
        img = np.zeros((img_height, img_width), dtype=np.uint8)
        points = np.array([list(zip(x.astype(int), y.astype(int)))])
        cv2.polylines(img, [points], isClosed=True, color=255, thickness=1)

        seed_x = int(np.mean(x)) #- 5
        seed_y = int(np.mean(y)) #- 5 
        mask = np.zeros((img_height + 2, img_width + 2), dtype=np.uint8)
        cv2.floodFill(img, mask, seedPoint=(seed_x, seed_y), newVal=255)
        max_length = max(len(x), len(y))

        x = np.pad(x, (0, max_length - len(x)), mode='edge')
        y = np.pad(y, (0, max_length - len(y)), mode='edge')
        coord = np.vstack((x, y)).T
        return torch.from_numpy(mask), torch.from_numpy(coord) 

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
            try:
                points = [tuple(map(float, line.split())) for line in lines]
            except:
                points = [eval(line) for line in lines]
            x, y = zip(*points)
            spline_pts = np.column_stack((x, y))

        label = int(self.labels[idx])
        if np.random.rand() < self.aug_prob:
            spline_pts = self.random_rotation(spline_pts)
        gt_mask, gt_coord = self.generate_gt(spline_pts, idx)
        return {
            "input": torch.tensor(spline_pts, dtype=torch.float),
            "mask": gt_mask,
            "coord": gt_coord
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

    def filter_points(self, x, y, threshold=5):
        x = x[x > threshold]
        y = y[y > threshold] 
        return x, y 

    def generate_gt(self, points, idx):  
        x = points[:, 0]
        y = points[:, 1]
        x, y = self.filter_points(x, y, 0)
        img_height = int(max(y)) + 10
        img_width = int(max(x)) + 10
        img = np.zeros((img_height, img_width), dtype=np.uint8)
        points = np.array([list(zip(x.astype(int), y.astype(int)))])
        cv2.polylines(img, [points], isClosed=True, color=255, thickness=1)

        seed_x = int(np.mean(x)) #- 5
        seed_y = int(np.mean(y)) #- 5 
        mask = np.zeros((img_height + 2, img_width + 2), dtype=np.uint8)
        cv2.floodFill(img, mask, seedPoint=(seed_x, seed_y), newVal=255)

        max_length = max(len(x), len(y))
        x = np.pad(x, (0, max_length - len(x)), mode='edge')
        y = np.pad(y, (0, max_length - len(y)), mode='edge')
        coord = np.vstack((x, y)).T
        return torch.from_numpy(mask), torch.from_numpy(coord) 
    
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
        x_filtered, y_filtered = self.filter_points(np.array(x), np.array(y), 1)
        max_length = max(len(x_filtered), len(y_filtered))
        x_filtered = np.pad(x_filtered, (0, max_length - len(x_filtered)), mode='edge')
        y_filtered = np.pad(y_filtered, (0, max_length - len(y_filtered)), mode='edge')
        padded_input = np.column_stack((x_filtered, y_filtered))
        gt_mask, gt_coord = self.generate_gt(spline_pts, idx)
        return {
            "input": torch.tensor(spline_pts, dtype=torch.float),
            "mask": gt_mask,
            "coord": torch.tensor(spline_pts, dtype=torch.float)
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

    def filter_points(self, x, y, threshold=5):
        x = x[x > threshold]
        y = y[y > threshold] 
        return x, y 

    def generate_gt(self, points, idx):  
        x = points[:, 0]
        y = points[:, 1]
        x, y = self.filter_points(x, y, 10)
        img_height = int(max(y)) + 10
        img_width = int(max(x)) + 10
        img = np.zeros((img_height, img_width), dtype=np.uint8)
        points = np.array([list(zip(x.astype(int), y.astype(int)))])
        cv2.polylines(img, [points], isClosed=True, color=255, thickness=1)

        seed_x = int(np.mean(x)) #- 5
        seed_y = int(np.mean(y)) #- 5 
        mask = np.zeros((img_height + 2, img_width + 2), dtype=np.uint8)
        cv2.floodFill(img, mask, seedPoint=(seed_x, seed_y), newVal=255)
        max_length = max(len(x), len(y))

        x = np.pad(x, (0, max_length - len(x)), mode='edge')
        y = np.pad(y, (0, max_length - len(y)), mode='edge')
        coord = np.vstack((x, y)).T
        return torch.from_numpy(mask), torch.from_numpy(coord) 
    
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
        gt_mask, gt_coord = self.generate_gt(spline_pts, idx)
        return {
            "input": torch.tensor(spline_pts, dtype=torch.float),
            "mask": gt_mask,
            "coord": gt_coord
        }
    
def collate_fn(batch):
    masks = [item["mask"] for item in batch]
    inputs = [item["input"] for item in batch]
    coords = [item["coord"] for item in batch]
    return {'mask': masks, 'input': inputs, 'coord': coords} 

def test_loader(): 
    print("----------- Testing Flavia -----------")
    path_flavia = "/home/salimkhazem/workspace/phd_thesis/experience_processing_time_contours/results_contours/data/Flavia/Flavia_sdp_None" 
    dataset = FlaviaDatasetSDP(path_flavia) 
    loader =  torch.utils.data.DataLoader(dataset, 16, collate_fn=collate_fn, shuffle=True)
    for data in loader: 
        inputs = data['input']
        masks = data['mask']  
        coords = data['coord']
        for i in range(len(inputs)):
            print(f"Sample {i}: Input: {inputs[i].shape} | Mask: {masks[i].shape} | Coord: {coords[i].shape}")
        break
    #plt.imshow(masks[5], cmap="gray")
    #plt.plot(inputs[5][:, 0], inputs[5][:, 1], "r")
    #plt.savefig(f"flavia_test.png", dpi=300)
    print("\n----------- Testing FashionMNIST -----------")
    path_fmnist = "/home/salimkhazem/workspace/phd_thesis/experience_processing_time_contours/results_contours/data/Fmnist/SIMPLE/train_sdp_SIMPLE" 
    dataset = FashionMNISTSDP(path_fmnist) 
    loader =  torch.utils.data.DataLoader(dataset, 16, collate_fn=collate_fn, shuffle=True)
    for data in loader: 
        inputs_ = data['input']
        masks_ = data['mask']  
        coords_ = data['coord']
        for i in range(len(inputs)):
            print(f"Sample {i}: Input: {inputs_[i].shape} | Mask: {masks_[i].shape} | Coord: {coords_[i].shape}")
        break
    plt.clf()
    plt.imshow(masks_[5], cmap="gray")
    plt.plot(inputs_[5][:, 0], inputs_[5][:, 1], "r")
    plt.savefig(f"Fmnist_test.png", dpi=300)



if __name__ == "__main__": 
    import matplotlib.pyplot as plt 
    test_loader()    





