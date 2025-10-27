import torch
import torch.nn as nn
# einops 库用于张量形状的灵活操作，如改变、重排等。
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F

# 从 segformer 导入相关模块，如 MixFFN, MLP_FFN, OverlapPatchEmbeddings
from networks.segformer import *


# 我的总结：
# 外部引入类：
# 根据您代码中的 from networks.segformer import * 语句以及后续的使用情况，您从 networks.segformer 模块中导入了以下三个类：
# 1.MixFFN：用于 CrossAttentionBlock、DualTransformerBlock 和 MyDecoderLayer 中的 MLP 层。
# 2.MLP_FFN：也用于 CrossAttentionBlock 和 DualTransformerBlock 中的 MLP 层。
# 3.OverlapPatchEmbeddings：用于 MiT 类中的初始特征嵌入，负责将输入图像转换为 Transformer 序列。

# 内部的类：
# 自定义类：
# Cross_Attention 
# CrossAttentionBlock
#  EfficientAttention ChannelAttention 
# DualTransformerBlock MiT（和另一包的mit是不同的,MIT也就是encoder） PatchExpand FinalPatchExpand_X4 
# MyDecoderLayer DAEFormer



# 我的第一个缝合模块：
class tongdao(nn.Module):  #处理通道部分   函数名就是拼音名称
    # 通道模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出大小为1x1
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于降维
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，就地操作以节省内存

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 提取批次大小和通道数
        y = self.avg_pool(x)  # 应用自适应平均池化
        y = self.fc(y)  # 应用1x1卷积
        y = self.relu(y)  # 应用ReLU激活
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  # 调整y的大小以匹配x的空间维度
        return x * y.expand_as(x)  # 将计算得到的通道权重应用到输入x上，实现特征重校准

class kongjian(nn.Module):
    # 空间模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于产生空间激励
        self.norm = nn.Sigmoid()  # Sigmoid函数用于归一化

    # 前向传播函数
    def forward(self, x):
        y = self.Conv1x1(x)  # 应用1x1卷积
        y = self.norm(y)  # 应用Sigmoid函数
        return x * y  # 将空间权重应用到输入x上，实现空间激励

class hebing(nn.Module):    #函数名为合并, 意思是把空间和通道分别提取的特征合并起来
    # 合并模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)  # 创建通道子模块
        self.kongjian = kongjian(in_channel)  # 创建空间子模块

    # 前向传播函数
    def forward(self, U):
        U_kongjian = self.kongjian(U)  # 通过空间模块处理输入U
        U_tongdao = self.tongdao(U)  # 通过通道模块处理输入U
        return torch.max(U_tongdao, U_kongjian)  # 取两者的逐元素最大值，结合通道和空间激励


class MDFA(nn.Module):                       ##多尺度空洞融合注意力模块。
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):# 初始化多尺度空洞卷积结构模块，dim_in和dim_out分别是输入和输出的通道数，rate是空洞率，bn_mom是批归一化的动量
        super(MDFA, self).__init__()
        self.branch1 = nn.Sequential(# 第一分支：使用1x1卷积，保持通道维度不变，不使用空洞
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential( # 第二分支：使用3x3卷积，空洞率为6，可以增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential( # 第三分支：使用3x3卷积，空洞率为12，进一步增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(# 第四分支：使用3x3卷积，空洞率为18，最大化感受野的扩展
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True) # 第五分支：全局特征提取，使用全局平均池化后的1x1卷积处理
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential( # 合并所有分支的输出，并通过1x1卷积降维
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.Hebing=hebing(in_channel=dim_out*5)# 整合通道和空间特征的合并模块

    def forward(self, x):
        [b, c, row, col] = x.size()
        # 应用各分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 全局特征提取
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # 合并所有特征
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 应用合并模块进行通道和空间特征增强
        larry=self.Hebing(feature_cat)
        larry_feature_cat=larry*feature_cat
        # 最终输出经过降维处理
        result = self.conv_cat(larry_feature_cat)

        return result





# --------------------------------------------------------------

# 跨注意力模块 (Cross_Attention)
# 这是一个自定义的跨注意力层，用于在两个不同的输入之间建立关联。
class Cross_Attention(nn.Module):
    # key_channels: K和Q的通道数；value_channels: V的通道数
    # height, width: 输入特征图的高度和宽度
    # head_count: 注意力头的数量
    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width

        # self.reprojection: 1x1 卷积层，用于将注意力输出的通道数翻倍。
        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        # self.norm: LayerNorm层，用于对重投影后的张量进行归一化。
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    # 前向传播函数
    def forward(self, x1, x2):
        # x1, x2 的形状都是 (B, N, D)，其中 N=H*W，D是嵌入维度
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        # 将输入张量从 (B, N, D) 重排为 (B, D, N)，以适应后续矩阵乘法
        # 注意：这里的queries和keys都来自x2，这似乎是一种特殊的跨注意力机制
        # 常见的是 queries 来自一个输入，keys和values来自另一个
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        # 计算每个注意力头的通道数
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        # 多头注意力循环
        for i in range(self.head_count):
            # 对 K 和 Q 进行 Softmax 操作，这与标准自注意力略有不同，可能是一种高效变体
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            # 提取当前头的值（V）
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            # 计算上下文向量 (context)，公式为 K * V^T
            context = key @ value.transpose(1, 2)  # dk*dv
            # 计算注意力加权值 (attended_value)，公式为 context^T * Q
            attended_value = context.transpose(1, 2) @ query  # n*dv
            # 将每个头的输出添加到列表中
            attended_values.append(attended_value)

        # 将所有头的输出在通道维度上拼接，并重塑回 (B, D, H, W)
        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        # 使用1x1卷积进行重投影，并展平为 (B, 2*D, N)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        # 进行 LayerNorm 归一化
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value


# 跨注意力块 (CrossAttentionBlock)
# 这是一个构建块，将Cross_Attention与MLP和残差连接结合
class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                    x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp="mix"):
        super().__init__()
        # 对 indim个通道进行层归一化
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        # 初始化 Cross_Attention 模块
        self.attn = Cross_Attention(key_dim, value_dim, height, width, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))
        # 根据 token_mlp 参数选择不同的 MLP 类型
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 对输入进行归一化
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)

        # 计算跨注意力
        attn = self.attn(norm_1, norm_2)
        # 将两个输入张量在通道维度上拼接，作为残差连接
        residual = torch.cat([x1, x2], dim=2)
        # 主路径 = 残差 + 注意力输出
        tx = residual + attn
        # 将 MLP 和第二个残差连接结合
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx


# 高效注意力模块 (EfficientAttention)
# 这是一个自注意力模块，用于单输入张量
class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]
    """
    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        
        # 定义用于生成 Q, K, V 的1x1卷积层
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        # 定义用于重投影的1x1卷积层
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        # 使用1x1卷积生成 Q, K, V，并展平为 (B, C, H*W)
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        # 计算每个头的通道数
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        # 多头自注意力循环
        for i in range(self.head_count):
            # 对 K 和 Q 进行 Softmax
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            # 提取当前头的值（V）
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            
            # 计算上下文向量 (context)
            context = key @ value.transpose(1, 2)  # dk*dv
            # 计算注意力加权值 (attended_value)，并重塑回 (B, C, H, W)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        # 将所有头的输出拼接
        aggregated_values = torch.cat(attended_values, dim=1)
        # 进行重投影
        attention = self.reprojection(aggregated_values)

        return attention


# 通道注意力模块 (ChannelAttention)
# 这是一个标准的通道注意力机制，与 Transformer 中的自注意力类似
class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        # temperature 是一个可学习参数，用于缩放注意力分数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.qkv: 线性层，用于生成 Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        # 使用 qkv 层生成 Q, K, V，并重塑为多头形式
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 转置 Q, K, V 以进行矩阵乘法
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # 对 Q 和 K 进行L2范数归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 计算注意力分数，并乘以可学习的温度参数
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        # 对注意力分数进行 softmax，得到注意力权重
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 将注意力权重与 V 相乘，得到加权后的值
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        # 使用 proj 层进行重投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# 双重注意力块 (DualTransformerBlock)
# 这是一个更复杂的构建块，结合了EfficientAttention和ChannelAttention
class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        # 初始化高效注意力模块
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        # 初始化通道注意力模块
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        # 根据 token_mlp 选择不同的 MLP
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # dual attention structure, efficient attention first then transpose attention
        # 双重注意力结构：先是高效注意力，然后是转置注意力（即通道注意力）

        # 对输入进行 LayerNorm
        norm1 = self.norm1(x)
        # 重排张量形状以适应 attn
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)

        # 执行高效注意力
        attn = self.attn(norm1)
        # 重排回 (B, N, D) 形状
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        # 第一个残差连接
        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        # 第二个残差连接
        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        # 执行通道注意力
        channel_attn = self.channel_attn(norm3)

        # 第三个残差连接
        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)

        # 最终输出
        mx = add3 + mlp2
        return mx


# 编码器 (MiT)
# 这部分是模型的骨干网络，负责特征提取
class MiT(nn.Module):
    #layers = [2, 2, 2]
    # in_dim = [128, 320,512]  key_dim = [128, 320,512],  value_dim = [128, 320,512]
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]

        # patch_embed
        # 定义分阶段的 OverlapPatchEmbeddings 模块，输入(1, 3, 224, 224)--》》(1, 128, 56, 56)。
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )

        # stage 1: in_dim=128, image_size=224  是经过了第一层的输出维度 和 进入第一层之前的高宽

        # stage 2: in_dim=320, image.size=56

        # stage 3: in_dim=512, image_size=28

        # transformer encoder
        # 定义多阶段的 DualTransformerBlock 模块
        self.block1 = nn.ModuleList(  #layers = (2, 2, 2) in_dim[0]=128
            [DualTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList( # in_dim[1]=320
            [DualTransformerBlock(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [DualTransformerBlock(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])
        # == 我自己缝合的模块 ==

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x) # 嵌入并下采样 (1, 3, 224, 224) ———》x = (1, 128, 56, 56)，h w 是56
        for blk in self.block1:
            x = blk(x, H, W) # 经过 transformer 块   ====  self.block1：经过 2 个 DualTransformerBlock 模块。每个模块内部会进行高效注意力、MLP 等操作，但张量形状保持不变。
        x = self.norm1(x) # 归一化
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # 重塑为 (B, C, H, W)
        outs.append(x) # 存储第一阶段的输出

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x) # 存储第二阶段的输出

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x) # 存储第三阶段的输出

        return outs # 返回所有阶段的输出，用于跳跃连接


# 解码器 (Decoder)
# 定义 PatchExpand 模块，用于上采样  该函数的效果是：将高宽各自乘以2，通道数除以4
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # 使用 rearrange 进行上采样和通道数压缩
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


# 最终的 PatchExpand 模块，上采样4倍
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        # 线性层，用于通道扩展，这里是 4x4=16 倍
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # rearrange 进行上采样
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale**2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


# 解码器层 (MyDecoderLayer)
# 这是一个编-解码器架构中的一个单元，处理一个解码阶段
class MyDecoderLayer(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            # 初始化跨注意力块
            self.cross_attn = CrossAttentionBlock(
                dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            # 最后一个解码层包含一个1x1卷积来预测最终的分割图
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
        
        # 两个 DualTransformerBlock 用于进一步的特征融合
        self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        
        # 权重初始化
        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            x1_expand = self.x1_linear(x1)
            # 交叉注意力，融合来自解码器主路径的 x1 和跳跃连接的 x2
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))
            # 经过两个 transformer 块
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                # 最终层进行上采样和卷积
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                # 中间层只进行上采样
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


# DAEFormer 模型
# 这是一个完整的编-解码器模型
class DAEFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        # 定义编码器参数
        dims, key_dim, value_dim, layers = [[128, 320, 512], [128, 320, 512], [128, 320, 512], [2, 2, 2]]
        self.backbone = MiT(
            image_size=224,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
        )

        # Decoder
        # 定义解码器参数
        # d_base_feat_size  空间分辨率基准
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        # p1 (输入通道)：该解码器阶段主要输入的通道数，通常来自上一个解码器阶段上采样后的结果。
        # p2 (编码器连接通道)：来自编码器对应阶段的跳跃连接的通道数，用于特征融合。
        # p3 / p4 (内部通道)：用于定义解码器层内部处理（例如注意力机制或 MLP）的中间通道数。
        # p5 (输出通道)：该解码器阶段最终输出的通道数，将作为下一个解码器阶段的输入。
        # p5 (输出通道)：该解码器阶段最终输出的通道数，将作为下一个解码器阶段的输入。
        in_out_chan = [
            [64, 128, 128, 128, 160],
            [320, 320, 320, 320, 256],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        # 初始化三个解码器层
        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x):
        # ---------------Encoder-------------------------
        # 如果输入是单通道，复制为三通道
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 经过编码器，得到不同尺度的特征图
        output_enc = self.backbone(x)

        b, c, _, _ = output_enc[2].shape

        # ---------------Decoder-------------------------
        # 解码器从最深层开始
        # decoder_2 处理编码器第三阶段的输出
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        # decoder_1 处理 decoder_2 的输出，并结合编码器第二阶段的输出（跳跃连接）
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        # decoder_0 处理 decoder_1 的输出，并结合编码器第一阶段的输出（跳跃连接）
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))

        # 返回最终的预测结果
        return tmp_0





        

# ### 二、前向传播模拟计算与讲解

# 让我们以一个批次大小为 `B=1`，输入图像大小为 `224x224` 的三通道图像为例，模拟 DAEFormer 模型的前向传播过程。

# #### 1. 编码器 (MiT)

# 输入张量 `x` 的形状为：`[1, 3, 224, 224]`

# **Stage 1：**

# * **Patch Embedding (`self.patch_embed1`)**: `OverlapPatchEmbeddings` 使用步长为4的卷积，将输入图像下采样4倍。
#     * 输出特征图尺寸 `H`, `W`： `ceil(224 / 4) = 56`。
#     * 输出通道数：`in_dim[0]` = 128。
#     * 输出张量 `x` 的形状：`[1, 56*56, 128]`。
# * **Dual Transformer Blocks (`self.block1`)**: 经过2个 `DualTransformerBlock`，特征图的形状和通道数保持不变。
# * **输出 (`outs[0]`)**: `x` 被重塑为 `[1, 128, 56, 56]` 并添加到 `outs` 列表中，作为解码器的跳跃连接。

# **Stage 2：**

# * **Patch Embedding (`self.patch_embed2`)**: 接收 `[1, 128, 56, 56]` 作为输入，继续下采样2倍。
#     * 输出特征图尺寸 `H`, `W`： `ceil(56 / 2) = 28`。
#     * 输出通道数：`in_dim[1]` = 320。
#     * 输出张量 `x` 的形状：`[1, 28*28, 320]`。
# * **Dual Transformer Blocks (`self.block2`)**: 经过2个 `DualTransformerBlock`，特征图的形状和通道数保持不变。
# * **输出 (`outs[1]`)**: `x` 被重塑为 `[1, 320, 28, 28]` 并添加到 `outs` 列表中，作为解码器的跳跃连接。

# **Stage 3：**

# * **Patch Embedding (`self.patch_embed3`)**: 接收 `[1, 320, 28, 28]` 作为输入，继续下采样2倍。
#     * 输出特征图尺寸 `H`, `W`： `ceil(28 / 2) = 14`。
#     * 输出通道数：`in_dim[2]` = 512。
#     * 输出张量 `x` 的形状：`[1, 14*14, 512]`。
# * **Dual Transformer Blocks (`self.block3`)**: 经过2个 `DualTransformerBlock`，形状和通道数保持不变。
# * **输出 (`outs[2]`)**: `x` 被重塑为 `[1, 512, 14, 14]` 并添加到 `outs` 列表中，作为解码器的跳跃连接。

# #### 2. 解码器 (Decoder)

# **解码器处理流程从最深层开始，即从 `output_enc[2]`（尺寸最小）开始。**

# **Decoder 2：**

# * **输入**: `output_enc[2]`，形状为 `[1, 512, 14, 14]`。
# * **操作**:
#     * 输入被展平为 `[1, 14*14, 512]`。
#     * `self.decoder_2` 中的 `PatchExpand` 模块对其进行2倍上采样。
#     * 上采样后的张量形状为 `[1, 28*28, 320]`。
# * **输出**: `tmp_2`，形状为 `[1, 28*28, 320]`。

# **Decoder 1：**

# * **输入**:
#     * 主路径输入 `x1`：`tmp_2`，形状为 `[1, 28*28, 320]`。
#     * 跳跃连接输入 `x2`： `output_enc[1]`，形状为 `[1, 320, 28, 28]`。
# * **操作**:
#     * `x2` 被展平为 `[1, 28*28, 320]`。
#     * `self.cross_attn` 模块融合 `x1` 和 `x2` 的特征。
#     * 融合后的特征经过 `DualTransformerBlock`。
#     * `self.decoder_1` 中的 `PatchExpand` 对融合后的特征进行2倍上采样。
#     * 上采样后的张量形状为 `[1, 56*56, 128]`。
# * **输出**: `tmp_1`，形状为 `[1, 56*56, 128]`。

# **Decoder 0（最终层）：**

# * **输入**:
#     * 主路径输入 `x1`：`tmp_1`，形状为 `[1, 56*56, 128]`。
#     * 跳跃连接输入 `x2`：`output_enc[0]`，形状为 `[1, 128, 56, 56]`。
# * **操作**:
#     * `x2` 被展平为 `[1, 56*56, 128]`。
#     * `self.cross_attn` 模块融合 `x1` 和 `x2` 的特征。
#     * 融合后的特征经过 `DualTransformerBlock`。
#     * `self.decoder_0` 中的 `FinalPatchExpand_X4` 对其进行4倍上采样。
#     * 上采样后的张量形状为 `[1, 224*224, 64]`。
#     * 最后，`self.last_layer` (一个1x1卷积) 将通道数从 `64` 映射到 `num_classes` (这里为9)。
# * **输出**: `tmp_0`，最终分割图的形状为 `[1, 9,
# 整个过程展示了编码器如何逐步下采样并提取高维特征，而解码器如何通过跳跃连接（skip connections）逐步上采样，同时融合来自编码器不同层级的特征，最终生成与原始输入图像分辨率相同的分割预测图。 224, 224]`。
