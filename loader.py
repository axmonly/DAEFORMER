from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from einops.layers.torch import Rearrange # 从einops库导入Rearrange模块，用于张量维度的灵活重排
from scipy.ndimage.morphology import binary_dilation # 从scipy库导入binary_dilation，用于对二值图像进行形态学膨胀操作，但此代码中未被使用

# ===== normalize over the dataset (对整个数据集进行标准化)
# 该函数对输入的图像数据集进行两次标准化操作：全局的Z-Score标准化和逐图像的Min-Max归一化。
def dataset_normalized(imgs):
    # imgs (numpy.ndarray): 输入的图像数组，形状通常为 (N, H, W, C)，其中N为样本数，H为高度，W为宽度，C为通道数。
    
    # 创建一个空的NumPy数组imgs_normalized，其形状与输入imgs完全相同。
    # np.empty() 仅分配内存，不进行初始化，因此性能最优，因为它后续会被计算结果完全覆盖。
    imgs_normalized = np.empty(imgs.shape) 
    
    # 计算整个数据集imgs的全局标准差。
    # np.std(imgs) 的输出是一个浮点数。
    imgs_std = np.std(imgs) 
    
    # 计算整个数据集imgs的全局均值。
    # np.mean(imgs) 的输出是一个浮点数。
    imgs_mean = np.mean(imgs) 
    
    # 执行全局Z-Score标准化：(x - mean) / std。
    # 这一步将所有像素值转换为均值为0，标准差为1的浮点数，存储在 imgs_normalized 中。
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    
    # 开始循环，对每一张图像（imgs.shape[0]是样本总数）执行逐图像的归一化。
    for i in range(imgs.shape[0]):
        # 对第i张图像进行Min-Max归一化，并将值映射到 [0, 255] 范围。
        # np.min(imgs_normalized[i]) 和 np.max(imgs_normalized[i]) 分别是第i张图像的最小值和最大值。
        # 归一化公式为 ((x - min) / (max - min)) * 255。
        # 这一步会覆盖掉之前的全局标准化结果，将每张图的像素值强制拉伸到 [0, 255] 范围，可能会抹平不同图像间的全局亮度差异。
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    
    # 返回最终处理后的图像数组，形状为 (N, H, W, C)，数据类型通常为浮点数。
    return imgs_normalized


## Temporary
class isic_loader(Dataset):
    """ 
    dataset class for Brats datasets.
    这是一个自定义的数据集类，它继承了PyTorch的Dataset类。
    它用于加载ISIC数据集，并进行必要的预处理和数据增强。
    """
    
    # 类的构造函数，用于初始化数据集。
    def __init__(self, path_Data, train = True, Test = False):
        # path_Data (str): 存储.npy数据文件的路径。
        # train (bool): 标志是否加载训练集。
        # Test (bool): 标志是否加载测试集。如果train为False且Test为False，则加载验证集。
        
        # 调用父类Dataset的构造函数。super()的调用方式在Python 3中可以简化为 super().__init__()。
        super(isic_loader, self)
        
        # 将train标志存储为实例变量。
        self.train = train
        
        # 根据train和Test标志，加载对应的数据集文件。
        if train:
            self.data  = np.load(path_Data+'data_train.npy') # 加载训练图像数据，形状为 (N, H, W, C)。
            self.mask  = np.load(path_Data+'mask_train.npy') # 加载训练掩码数据，形状为 (N, H, W)。
        else:
            if Test:
                self.data  = np.load(path_Data+'data_test.npy') # 加载测试图像数据，形状为 (N, H, W, C)。
                self.mask  = np.load(path_Data+'mask_test.npy') # 加载测试掩码数据，形状为 (N, H, W)。
            else:
                self.data  = np.load(path_Data+'data_val.npy') # 加载验证图像数据，形状为 (N, H, W, C)。
                self.mask  = np.load(path_Data+'mask_val.npy') # 加载验证掩码数据，形状为 (N, H, W)。
        
        # 对加载的图像数据进行标准化处理，调用上面定义的函数。
        # 输入形状为 (N, H, W, C)，输出形状保持不变，但像素值已被处理。
        self.data  = dataset_normalized(self.data)
        
        # 使用np.expand_dims给掩码数据增加一个通道维度。
        # 输入形状为 (N, H, W)，输出形状变为 (N, H, W, 1)。
        self.mask  = np.expand_dims(self.mask, axis=3)
        
        # 将掩码像素值从 [0, 255] 范围归一化到 [0, 1] 的浮点数范围。
        self.mask  = self.mask /255.

    # 该方法是PyTorch Dataset类的核心，用于按索引获取单个数据样本。
    def __getitem__(self, indx):
        # indx (int): 样本的索引。
        
        # 从实例变量中获取指定索引的图像和掩码数据。
        # img 的形状为 (H, W, C)。
        img = self.data[indx] 
        # seg 的形状为 (H, W, 1)。
        seg = self.mask[indx]
        
        # 如果是训练模式，则应用数据增强。
        if self.train:
            # apply_augmentation的输入和输出形状都是 (H, W, C) 和 (H, W, 1)。
            img, seg = self.apply_augmentation(img, seg) 
        
        # 将NumPy数组seg转换为PyTorch张量。
        # .copy() 用于确保数据在转换前是连续的，以避免潜在的内存问题。
        seg = torch.tensor(seg.copy()) 
        # 将NumPy数组img转换为PyTorch张量。
        img = torch.tensor(img.copy()) 
        
        # 使用.permute()方法重排图像张量的维度。
        # 将图像从 (H, W, C) 的NumPy惯例重排为 (C, H, W) 的PyTorch惯例。
        # 转换后，img的形状为 (C, H, W)。
        img = img.permute( 2, 0, 1) 
        # 同样，重排掩码张量的维度。
        # 将掩码从 (H, W, 1) 变为 (1, H, W)。
        seg = seg.permute( 2, 0, 1) 

        # 返回一个包含图像和掩码张量的字典。
        # 'image' 的值是形状为 (C, H, W) 的张量。
        # 'mask' 的值是形状为 (1, H, W) 的张量。
        return {'image': img, 
                'mask' : seg}
                
    # 辅助方法，用于执行随机数据增强。
    def apply_augmentation(self, img, seg):
        # img (numpy.ndarray): 待增强的图像，形状为 (H, W, C)。
        # seg (numpy.ndarray): 待增强的掩码，形状为 (H, W, 1)。
        
        # 生成一个0到1之间的随机浮点数。
        if random.random() < 0.5:
            # 如果随机数小于0.5，则有50%的概率执行水平翻转。
            # np.flip() 沿着指定轴（axis=1，即宽度轴）翻转数组。
            img  = np.flip(img,  axis=1) # 翻转图像，形状仍为 (H, W, C)。
            seg  = np.flip(seg,  axis=1) # 翻转掩码，形状仍为 (H, W, 1)。
        # 返回翻转后的图像和掩码。
        return img, seg

    # 该方法返回数据集中的样本总数。
    def __len__(self):
        # 返回self.data（图像数组）的第一维大小，即样本数N。
        return len(self.data)












# ---

# ### 2. 带入代码进行模拟计算

# 让我们以一个具体的例子来模拟整个数据加载过程。

# **假设：**
# * `path_Data = './isic_data/'`
# * `train = True`
# * 数据集包含 **100** 张 `224x224` 的三通道图像 (`data_train.npy`)。
# * 数据集包含 **100** 张 `224x224` 的单通道掩码 (`mask_train.npy`)。

# #### **第一步：`__init__` 方法执行**

# 当您创建 `isic_loader` 实例时，构造函数会被调用。

# ```python
# isic_dataset = isic_loader(path_Data='./isic_data/', train=True)