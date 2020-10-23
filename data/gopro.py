import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from PIL import Image
from torchvision import transforms


# 测试语句
# image_blur, image_sharp = get_image_path(False)
# print(len(image_blur), len(image_sharp))
# image0 = Image.open(image_sharp[0])
# image1 = Image.open(image_blur[0])
# image0.show()
# width, height = image0.size
# print(width, height)

class GoProDataSet(Dataset):
    """
    加载数据
    """

    def __init__(self, img_list, crop_size=(256, 256)):
        super().__init__()
        self.img_list = img_list
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

    def crop_resize_totensor(self, img, crop_location):
        """
        :param img: 输入的图片
        :param crop_location: 图片
        :return: 三种尺度的图片
        """
        img256 = img.crop(crop_location)
        img128 = img256.resize((self.crop_size[0] // 2, self.crop_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((self.crop_size[0] // 4, self.crop_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __getitem__(self, index):
        blurry_img_path = self.img_list[index].split(' ')[-2]
        clear_img_path = self.img_list[index].split(' ')[-1]

        x_img = Image.open(blurry_img_path)
        y_img = Image.open(clear_img_path)

        assert x_img == y_img

        crop_left = int(np.floor(np.random.uniform(0, x_img.size[0] - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, x_img.size[1] - self.crop_size[1] + 1)))
        crop_location = (crop_left, crop_top, crop_left + self.crop_size[0], crop_top + self.crop_size[1])

        img256, img128, img64 = self.crop_resize_totensor(x_img, crop_location)
        label256, label128, label64 = self.crop_resize_totensor(y_img, crop_location)
        batch = {'img256': img256, 'img128': img128, 'img64': img64, 'label256': label256, 'label128': label128,
                 'label64': label64}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch

    def __len__(self):
        return len(self.img_list)


# img_dataset = GoProDataSet()
# train_loader = DataLoader(dataset=img_dataset, batch_size=10, shuffle=True)
# for i, data in enumerate(train_loader):
#     x, y = data
#     print(i, y.shape)

class TestDataSet(Dataset):
    def __init__(self, img_list):
        super().__init__()
        self.img_list = img_list
        self.to_tensor = transforms.ToTensor()

    def resize_totensor(self, img):
        img_size = img.size
        img256 = img
        img128 = img256.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)
        img64 = img128.resize((img_size[0] // 4, img_size[1] // 4), resample=Image.BILINEAR)
        return self.to_tensor(img256), self.to_tensor(img128), self.to_tensor(img64)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # filename processing
        blurry_img_name = self.img_list[idx].split(' ')[-2]
        clear_img_name = self.img_list[idx].split(' ')[-1]

        blurry_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)
        assert blurry_img.size == clear_img.size

        img256, img128, img64 = self.resize_totensor(blurry_img)
        label256 = self.to_tensor(clear_img)
        batch = {'img256': img256, 'img128': img128, 'img64': img64, 'label256': label256}
        for k in batch:
            batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
        return batch
