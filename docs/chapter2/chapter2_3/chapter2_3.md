# 基于 Transformers，diffusion 技术解析+实战

# 1、Transformers+diffusion 技术背景简介

## Transformers diffusion 背景

近期大火的 OpenAI 推出的 Sora 模型，其核心技术点之一，是将视觉数据转化为 Patch 的统一表示形式，并通过 Transformers 技术和扩散模型结合，展现了卓越的 scale 特性。

被 Twitter 上广泛传播的论文《Scalable diffusion models with transformers》也被认为是 Sora 技术背后的重要基础。而这项研究的发布遇到了一些坎坷，曾经被 CVPR2023 拒稿过。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/3BMqYa1z8WmvqwZL/img/6116a27d-77fa-47f5-a8f1-8ac0ba63c239.png)

然 DiT 被拒了，我们看到来自清华大学，人民大学等机构共同研究的 CVPR2023 的论文 U-ViT《All are Worth Words: A ViT Backbone for Diffusion Models》，2022 年 9 月发表，这项研究设计了一个简单而通用的基于 vit 的架构（U-ViT），替换了 U-Net 中的卷积神经网络（CNN），用于 diffusion 模型的图像生成任务。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/3BMqYa1z8WmvqwZL/img/1d4cbd10-3547-4b7c-9beb-13b53781611e.png)

该项研究现已开源，欢迎大家关注：

GitHub 链接： [https://github.com/baofff/U-ViT](https://github.com/baofff/U-ViT)

论文链接：[https://arxiv.org/abs/2209.12152](https://arxiv.org/abs/2209.12152)

模型链接：[https://modelscope.cn/models/thu-ml/imagenet256_uvit_huge](https://modelscope.cn/models/thu-ml/imagenet256_uvit_huge)

但是，正如作者所说的，Sora 将基于 Transformers 的 diffusion model scale up 成功，不仅需要对底层算法有专家级理解，还要对整个深度学习工程体系有很好的把握，这项工作相比在学术数据集做出一个可行架构更加困难。

## 什么是 ViT

Vision Transformer (ViT)  模型由  Alexey Dosovitskiy 等人在  An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  中提出。这是第一篇在  ImageNet  上成功训练  Transformer  编码器的论文，与熟悉的卷积架构相比，取得了非常好的结果。论文提出，虽然  Transformer  架构已成为自然语言处理任务事实上的标准，但其在计算机视觉中的应用仍然有限。  在视觉中，attention 要么与卷积网络结合应用，要么用于替换卷积网络的某些组件，同时保持其整体结构不变。 ViT 证明这种对  CNN  的依赖是不必要的，直接应用于图像块序列（patches）的纯  Transformer  可以在图像分类任务上表现良好。  当对大量数据进行预训练并转移到多个中型或小型图像识别基准（ImageNet、CIFAR-100、VTAB  等）时，Vision Transformer (ViT)  与 SOTA 的 CNN 相比取得了优异的结果，同时需要更少的计算资源来训练，Vision Transformer (ViT)  基本上是  Transformers，但应用于图像。

每个图像被分割成一系列不重叠的块（分辨率如  16x16  或  32x32），并线性 embedding，接下来，添加 position embedding，并通过编码器层发送。  在开头添加  \[CLS\]  标记以获得整个图像的表示。  可以在 hidden states 之上添加 MLP head 以对图像进行分类。

ViT 架构：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/MAeqxY8Qe9pKO8j9/img/1bbc54a3-9d11-4b16-b7d0-c08f5c61852d.png)

\-来自原论文：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

Paper: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

Official repo (in JAX): [https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)

## ViT 在大语言模型中的使用（Qwen-VL 为例）

- Qwen-VL: Qwen-VL  以  Qwen-7B  的预训练模型作为语言模型的初始化，并以 openclip-ViT-bigG 作为视觉编码器的初始化，中间加入单层随机初始化的  cross-attention，经过约 1.5B 的图文数据训练得到。最终图像输入分辨率为 448\*448。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1GXn4XpbrGgGnDQ4/img/ba7d6465-e93d-495b-9ad7-6c468175548f.png)

论文链接：

[https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

## ViViT：视频 ViT

ViViT 基于纯变压器的视频模型分类，借鉴了 ViT 图像分类中取得的成功。 ViViT 从输入视频中提取时空标记，然后由一系列转换器层进行编码。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/MAeqxY8Qe9pKO8j9/img/f9e2e3f7-dec9-4bea-a190-0fc738ed7cbc.png)

源自：Arnab, Anurag, et al. "Vivit: A video vision transformer." ICCV2021

paper：[https://arxiv.org/abs/2103.15691](https://arxiv.org/abs/2103.15691)

### Latte:用于视频生成的潜在扩散变压器

Latte 提出了一种新颖的潜在扩散变压器，用于视频生成。Latte  首先从输入视频中提取时空标记，然后采用一系列  Transformer  块对潜在空间中的视频分布进行建模。为了对从视频中提取的大量标记进行建模，从分解输入视频的空间和时间维度的角度引入了四种有效的变体。为了提高生成视频的质量，我们通过严格的实验分析确定了  Latte  的最佳实践，包括视频剪辑补丁嵌入、模型变体、时间步级信息注入、时间位置嵌入和学习策略。我们的综合评估表明，Latte  在四个标准视频生成数据集（即  FaceForensics、SkyTimelapse、UCF101  和  Taichi-HD）上实现了最先进的性能。此外， Latte 也   扩展到文本到视频生成  (T2V)  任务，其中  Latte  取得了与最新  T2V  模型相当的结果。我们坚信，Latte  为未来将  Transformer  纳入视频生成扩散模型的研究提供了宝贵的见解。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/MAeqxY8Qe9pKO8j9/img/165e292e-11cf-48f2-a173-a23dcc838cd8.png)

# 2、UViT 读论文（聂同学）

# 3、代码实战

[Patch 最佳实践](./patch-BestPractice.ipynb)

[ViT 最佳实践](./ViT-BestPractice.ipynb)

[UViT 最佳实践](./UViT_ImageNet_demo.ipynb)

[ViViT 最佳实践](./ViViT-BestPractice.ipynb)

[Latte 最佳实践](./Latte-BestPractice.ipynb)

# 4、录播回放地址

[录播回放](https://www.bilibili.com/video/BV1px421y7qU/?vd_source=79686b80ce91d6c3977b2e269db5e8b8)
