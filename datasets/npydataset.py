import torch
from torch.utils.data import Dataset
import numpy as np
import os

class NpyDataset(Dataset):
    def __init__(self, nums=400, root_dir='exp/image_samples/trainset_celeba', steps=['999']):
        """
        Args:
            root_dir (str): 数据存放的根目录，包含 0.npy 到 399.npy。
        """
        super(NpyDataset, self).__init__()
        self.steps = steps
        self.root_dir = root_dir
        self.num = nums
        self.file_list = [f"{i}.npy" for i in range(nums)]  # 文件名列表

    def __len__(self):
        """返回数据集的大小。"""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        根据索引加载对应的 .npy 文件，并返回数据。

        Args:
            idx (int): 数据索引。

        Returns:
            torch.Tensor: 加载的 numpy 数据，转换为 PyTorch 张量。
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        data = []
        for step in self.steps:
            if step == self.steps[-1]:
                file_path = os.path.join(self.root_dir, step, 'x0_' + self.file_list[idx])
            else:
                file_path = os.path.join(self.root_dir, step, 'x0_pred_' + self.file_list[idx])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist.")

            # 加载 .npy 文件
            data.append(np.expand_dims(np.load(file_path), axis=0))
        data = np.vstack(data)
        gt = np.load(os.path.join(self.root_dir, 'orig', self.file_list[idx]))
        y_0 = np.load(os.path.join(self.root_dir, 'y_0', 'y0_'+self.file_list[idx]))

        # 将 numpy 数组转换为 PyTorch 张量
        return torch.tensor(data, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32), torch.tensor(y_0, dtype=torch.float32)


class NpyDataset_cache(Dataset):
    def __init__(self, nums=400, root_dir='exp/image_samples/trainset_celeba', steps=['999']):
        """
        Args:
            root_dir (str): 数据存放的根目录，包含 0.npy 到 399.npy。
        """
        super(NpyDataset_cache, self).__init__()
        self.steps = steps
        self.root_dir = root_dir
        self.num = nums
        self.file_list = [f"{i}.npy" for i in range(nums)]  # 文件名列表
        self._read_all_data()

    def _read_all_data(self):
        self.data_list = []
        self.gt_list = []
        self.y_0_list = []
        for idx in range(self.__len__()):
            data = []
            for step in self.steps:
                if step == self.steps[-1]:
                    file_path = os.path.join(self.root_dir, step, 'x0_' + self.file_list[idx])
                else:
                    file_path = os.path.join(self.root_dir, step, 'x0_pred_' + self.file_list[idx])
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File {file_path} does not exist.")

                # 加载 .npy 文件
                data.append(np.expand_dims(np.load(file_path), axis=0))
            data = np.vstack(data)
            gt = np.load(os.path.join(self.root_dir, 'orig', self.file_list[idx]))
            y_0 = np.load(os.path.join(self.root_dir, 'y_0', 'y0_'+self.file_list[idx]))
            self.data_list.append(data)
            self.gt_list.append(gt)
            self.y_0_list.append(y_0)
        # return
        self.data_list = np.stack(self.data_list, axis=0)
        self.gt_list = np.stack(self.gt_list, axis=0)
        self.y_0_list = np.stack(self.y_0_list, axis=0)

    def get_all_data(self):
        return torch.tensor(self.data_list, dtype=torch.float32), torch.tensor(self.gt_list, dtype=torch.float32), torch.tensor(self.y_0_list, dtype=torch.float32)

    def __len__(self):
        """返回数据集的大小。"""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        根据索引加载对应的 .npy 文件，并返回数据。

        Args:
            idx (int): 数据索引。

        Returns:
            torch.Tensor: 加载的 numpy 数据，转换为 PyTorch 张量。
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        data = self.data_list[idx]
        gt = self.gt_list[idx]
        y_0 = self.y_0_list[idx]

        # 将 numpy 数组转换为 PyTorch 张量
        return torch.tensor(data, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32), torch.tensor(y_0, dtype=torch.float32)

# 使用示例
if __name__ == "__main__":
    root_dir = "path/to/npy/files"  # 替换为你的数据路径
    dataset = NpyDataset(root_dir=root_dir)

    # 打印数据集大小
    print(f"Dataset size: {len(dataset)}")

    # 加载第 0 个样本
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")

    # 使用 DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx} shape: {batch.shape}")
        break
