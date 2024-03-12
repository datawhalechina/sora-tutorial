# 训练一个sora模型的准备工作，video caption和算力评估
<!-- 我是正文 -->
# 如何开始训练一个Sora模型

**训练Sora模型**

在Sora的技术报告中，Sora使用视频压缩网络将各种大小的视频压缩为潜在空间中的时空patches sequence，然后使用Diffusion Transformer进行去噪，最后解码生成视频。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/f13cc86d-6029-448b-be9c-5e8e14f78b97.png)

Open-Sora 在下图中总结了 Sora 可能使用的训练流程。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/314abd6f-6837-4a06-baad-8578f2b977c9.png)

图片来源：[https://hpc-ai.com/blog/open-sora](https://hpc-ai.com/blog/open-sora)

训练链路：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/e0ef2f61-5acb-49f6-bc1f-4015d590458d.png)

**数据准备**

开源数据集：

**VideoInstruct-100K：**

VideoInstruct100K 是使用人工辅助和半自动注释技术生成的高质量视频对话数据集。数据集中的问题答案与以下内容相关：

*   视频摘要
    
*   基于描述的问题答案（探索空间、时间、关系和推理概念）
    
*   创意/生成性问题解答
    

链接：[https://modelscope.cn/datasets/AI-ModelScope/VideoInstruct-100K](https://modelscope.cn/datasets/AI-ModelScope/VideoInstruct-100K/summary)

**panda-70m：**

Panda-70M 是一个包含 70M 高质量视频字幕对的大规模数据集。该存储库分为三个部分：

*   数据集数据加载包括列出 Panda-70M 数据的 csv 文件以及下载数据集的代码。
    
*   分割包括将长视频分割成多个语义一致的短片的代码。
    
*   字幕包括在 Panda-70M 上训练的拟议视频字幕模型。
    

**链接：**[https://modelscope.cn/datasets/AI-ModelScope/panda-70m](https://modelscope.cn/datasets/AI-ModelScope/panda-70m/summary)

**Youku-mPLUG:**

Youku-mPLUG预训练数据集挖掘自优酷站内海量的优质短视频内容

*   包含千万级别约36TB的视频、文本数据。
    
*   其中视频均为覆盖10～120秒的UGC短视频内容，文本为视频对应的描述标题，长度5～30不等。
    
*   该数据集抽取时品类均衡，内容共包含45个大类。
    

链接：[https://modelscope.cn/datasets/modelscope/Youku-AliceMind](https://modelscope.cn/datasets/modelscope/Youku-AliceMind/summary)

**MSR-VTT：**

MSR-VTT（Microsoft Research Video to Text）是一个开放域视频字幕的大规模数据集。

*   由 20 个类别的 10,000 个视频片段组成，每个视频片段由 Amazon Mechanical Turks 标注了 20 个英文句子。
    
*    所有标题中约有 29,000 个独特单词。 
    
*   标准分割使用 6,513 个split用于训练，497 个split用于验证，2,990 个split用于测试。
    

链接：[https://modelscope.cn/datasets/AI-ModelScope/msr-vtt](https://modelscope.cn/datasets/AI-ModelScope/msr-vtt/summary)

**Shot2Story：**

视频文本基准和用于多镜头视频理解的可扩展代码。包含20k 视频的详细长摘要和 80k 视频镜头的镜头字幕。

链接：[https://modelscope.cn/datasets/AI-ModelScope/Shot2Story](https://modelscope.cn/datasets/AI-ModelScope/Shot2Story/summary)

**InternVid：**

InternVid 是一个以视频为中心的大规模多模态数据集，可以学习强大且可转移的视频文本表示，以实现多模态理解和生成。 InternVid 数据集包含超过 700 万个视频，持续近 76 万小时，产生 2.34 亿个视频剪辑，并附有总共 4.1B 个单词的详细描述。

链接：[https://modelscope.cn/datasets/AI-ModelScope/InternVid](https://modelscope.cn/datasets/AI-ModelScope/InternVid/summary)

**webvid-10M：**

大型文本视频数据集，包含从素材网站抓取的**1000 万个视频文本对。**

链接：[https://modelscope.cn/datasets/AI-ModelScope/webvid-10M](https://modelscope.cn/datasets/AI-ModelScope/webvid-10M/summary)

**数据预处理**

目前主流LLM框架缺乏针对 video数据 统一便捷的管理和处理能力，且多模态数据处理标准方案缺失

*   Huggingface-Datasets 官方认为video比image更棘手，[暂未支持](https://discuss.huggingface.co/t/hf-dataset-for-videos/45900)
    
*   相关video库对该场景过于庞杂或简单
    

*   [FFmpeg](https://github.com/FFmpeg/FFmpeg)：150w行+源码，大量底层细节
    
    *   pytorchvideo：主要支持[加载](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html)和少量单video模态的[tensor transform](https://pytorchvideo.readthedocs.io/en/latest/api/transforms/transforms.html#module-pytorchvideo.transforms.functional)（翻转、扰动、采样等）
        
*   SORA官方仅模糊提及使用了DALLE3来生成caption，细粒度的"caption --> spacetime patch"建模比较关键
    
*   从SORA模型效果看，数据需要有变化的时长、分辨率和宽高比
    

[Data-Juicer](https://github.com/alibaba/data-juicer/docs/DJ_SORA_ZH.md) 扩展了对多模态数据的支持，已实现上百个专用的视频、图像、音频、文本等多模态数据处理算子及工具，帮助用户分析、清洗及生成大规模高质量数据。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/9554c372-caba-4c36-94c0-46499400da04.png)

*   支持视频数据的高性能IO和处理
    

*   支持并行化数据加载：lazy load with pyAV and ffmpeg；多模态数据路径签名
    
    *   并行化算子处理：支持单机多核；GPU调用；Ray多机分布式
        
    *   \[WIP\] 分布式调度优化；分布式存储优化
        

*   基础算子（视频时空维度）
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/bab05fbd-b589-41de-a268-f7f19583e9f1.png)

*   基础算子（细粒度模态间匹配及生成）
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/88127ccf-b7da-4567-806e-3c572db3ba9d.png)

*   进阶算子（视频内容）
    

          ![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/0c5d9fcf-603a-4407-9c3a-6ee76563a85c.png)

*   DJ-SORA数据菜谱及数据集
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/3be5786e-e82e-49bc-98c8-177f1023e98f.png)

*   DJ-SORA数据验证及模型训练
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/f54377ed-4ae9-443f-8cee-04059cf9ec5d.png)

（开源链接：[https://github.com/alibaba/data-juicer/docs/DJ\_SORA\_ZH.md](https://github.com/alibaba/data-juicer/docs/DJ_SORA_ZH.md)

**近期将开展详细的DJ-SORA和Data-Juicer系统技术分享，敬请期待！****）**

**模型选型和训练**

## 视频VQVAE

VideoGPT 使用 VQ-VAE，通过采用 3D 卷积和轴向自注意力来学习原始视频的下采样离散潜在表示。然后使用一个简单的类似 GPT 的架构，使用时空位置编码对离散潜在变量进行自回归建模。用于 BAIR Robot 数据集上的视频生成，并从 UCF-101 和 Tumbler GIF 生成高保真自然图像数据集（TGIF）。

[https://github.com/wilson1yan/VideoGPT/](https://github.com/wilson1yan/VideoGPT/)

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/ed0fd2a2-a22f-48a7-8114-7236eab8a56c.png)

## Diffusion Transformer

普遍认为Diffusion Transformer模型是Sora的技术基础，通过结合diffusion model和transformer，从而达到可以scale up model来提升图像生成质量的效果。我们总结了三个目前开源的Diffusion Transformer研究如下，并总结了最佳实践，可以在魔搭社区的免费算力上运行和测试。

**UViT：**All are Worth Words: A ViT Backbone for Diffusion Models

论文链接：[https://arxiv.org/abs/2209.12152](https://arxiv.org/abs/2209.12152)

代码库链接：[https://github.com/baofff/U-ViT](https://github.com/baofff/U-ViT)

模型链接：[https://modelscope.cn/models/thu-ml/imagenet256\_uvit\_huge](https://modelscope.cn/models/thu-ml/imagenet256_uvit_huge/summary)

最佳实践：[https://github.com/modelscope/modelscope/blob/master/examples/pytorch/UViT\_ImageNet\_demo.ipynb](https://github.com/modelscope/modelscope/blob/master/examples/pytorch/UViT_ImageNet_demo.ipynb)

效果图：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Pd6l2YoAj7rmO7Ma/img/35e64204-22f3-461b-90e5-361d0965a3bb.png)![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Pd6l2YoAj7rmO7Ma/img/2d2437b5-229d-42cb-b9af-e09ae9d46f1d.png)

**DiT：**Scalable Diffusion Models with Transformers

论文链接：[https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)

代码库链接：[https://github.com/facebookresearch/DiT](https://github.com/facebookresearch/DiT)

模型链接：[https://modelscope.cn/models/AI-ModelScope/DiT-XL-2-256x256/summary](https://modelscope.cn/models/AI-ModelScope/DiT-XL-2-256x256/summary)

最佳实践：[https://github.com/modelscope/modelscope/blob/master/examples/pytorch/DiT\_ImageNet\_Demo.ipynb](https://github.com/modelscope/modelscope/blob/master/examples/pytorch/DiT_ImageNet_Demo.ipynb)

效果图：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Pd6l2YoAj7rmO7Ma/img/8c6bf338-3acf-4363-bddd-6b333548f44a.png)

**SiT：**Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (SiT)

论文链接：[https://arxiv.org/pdf/2401.08740.pdf](https://arxiv.org/pdf/2401.08740.pdf)

代码库链接：[https://github.com/willisma/SiT](https://github.com/willisma/SiT)

模型链接：[https://modelscope.cn/models/AI-ModelScope/SiT-XL-2-256](https://modelscope.cn/models/AI-ModelScope/SiT-XL-2-256/summary)

最佳实践：[https://github.com/modelscope/modelscope/blob/master/examples/pytorch/SiT\_ImageNet\_Demo.ipynb](https://github.com/modelscope/modelscope/blob/master/examples/pytorch/SiT_ImageNet_Demo.ipynb)

效果图：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Pd6l2YoAj7rmO7Ma/img/366b3def-5c2f-40a2-bec5-52484f0a8dd3.png)

总结对比（部分观点来自知乎：[https://zhuanlan.zhihu.com/p/619033826?utm\_psn=1743677564626051072](https://zhuanlan.zhihu.com/p/619033826?utm_psn=1743677564626051072)）

**U-ViT**是一种简单且通用的基于ViT的扩散概率模型的主干网络，U-ViT把所有输入，包括图片、时间、条件都当作token输入，并且引入了**long skip connection**。U-ViT在无条件生成、类别条件生成以及文到图生成上均取得了可比或者优于CNN的结果。为未来扩散模型中骨干网络研究提供见解，并有利于大规模跨模态数据集的生成建模。

**DiT**同样的提出了使用ViT代替U-Net的思想，不同的是DiT中没有引入long skip connection也依然取得了杰出的效果。推测原因可能有：

*   DiT 出色的**Adaptive layer norm**以及**零初始化的设计**能够有效提升生成质量；
    
*   DiT 在建模特征空间表现良好，但在建模像素空间表现欠缺，可能在用扩散概率模型建模像素空间分布时long skip connection是至关重要的；
    
*   即使在建模特征空间上，DiT 没有long skip connection也能取得很好的效果，但long skip connection在加速收敛方面也起着关键的作用。
    

而近期推出的可扩展插值变压器 **(SiT)**，是建立在DiT 基础上的生成模型系列。 **插值框架，**相比标准的diffusion模型允许以更灵活的方式连接两个distributions，使得对影响生成的各种设计选择的模块化研究成为可能。SiT 在 ImageNet 256x256 基准上模型大小和效果超过了 DiT和UViT，SiT 实现了 **2.06 的 FID-50K 分数。**

**开发者的一个问题：**

想提一个直播希望解答的问题：阅读了Stable Diffusion 3的论文之后，感觉其重点强调的Rectified Flow方法，就是SiT论文中的General Interpolants里面的Linear方案，同时也是用的基于速度估计的扩散模型（不同于DDPM和基于Score的扩散模型），在复现Sora时，由于Stable Diffustion 3显示出的强大性能但是暂不开源，是否选择SiT比OpenDiT更合理一些？

答复：我感觉不是重点。。loss感觉就是探索了不同的超参数：noise schedule里的alpha/sigma，不同t的权重，训练时t的采样方式，以及不同的prediction方式（eps / v / edm / rf），还是diffusion那一套，好像没必要用上他提到的flow matching这些东西。就是loss那些改不改是锦上添花的事情，主要解决的还是 可变分辨率/时长、long context、scaling transformer这些

## Video-caption

OpenAI训练了一个具备高度描述性的视频标题生成（Video Captioning）模型，使用这个模型为所有的视频训练数据生成了高质量文本标题，再将视频和高质量标题作为视频文本对进行训练。通过这样的高质量的训练数据，保障了文本（prompt）和视频数据之间高度的align。通过近期的讨论和资料，我们推测Video Captioning模型是由多模态大语言模型VLM（如**GPT4V模型）**微调出来的。开发者也可以通过视频抽帧+开源VLM生成描述+LLM总结描述的方式，生成较好的视频描述。下面是一些开源的多模态模型：

**零一万物VL模型（****Yi-VL-34B****）**

代码库链接：[https://github.com/01-ai/Yi/tree/main/VL](https://github.com/01-ai/Yi/tree/main/VL)

模型链接：[https://modelscope.cn/models/01ai/Yi-VL-34B/](https://modelscope.cn/models/01ai/Yi-VL-34B/summary)

最佳实践：[https://mp.weixin.qq.com/s?\_\_biz=MzkxNTM5NTg2OA==&mid=2247488964&idx=1&sn=b40140340c9ce817f8181e94a221dbc8&chksm=c15e91b7f62918a178419556b3099ee4cff617936a7197b97c9409978613cb4cabfe9b70d609&token=740682062&lang=zh\_CN#rd](https://mp.weixin.qq.com/s?__biz=MzkxNTM5NTg2OA==&mid=2247488964&idx=1&sn=b40140340c9ce817f8181e94a221dbc8&chksm=c15e91b7f62918a178419556b3099ee4cff617936a7197b97c9409978613cb4cabfe9b70d609&token=740682062&lang=zh_CN#rd)

**通义千问VL模型（Qwen-VL-Chat）**

论文链接：[https://arxiv.org/abs/2308.12966](https://arxiv.org/abs/2308.12966)

代码库链接：[https://github.com/QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL)

模型链接：[https://modelscope.cn/models/qwen/Qwen-VL-Chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary)

最佳实践：[https://mp.weixin.qq.com/s?\_\_biz=MzkxNTM5NTg2OA==&mid=2247486027&idx=1&sn=4fd6c93edd5a97ec017d692b4bc81506&chksm=c15e8e38f629072e0df1eb4d00b2d71281f1fd77b2c138ba95a007643d0697b6a8f7eb697ddd&token=740682062&lang=zh\_CN#rd](https://mp.weixin.qq.com/s?__biz=MzkxNTM5NTg2OA==&mid=2247486027&idx=1&sn=4fd6c93edd5a97ec017d692b4bc81506&chksm=c15e8e38f629072e0df1eb4d00b2d71281f1fd77b2c138ba95a007643d0697b6a8f7eb697ddd&token=740682062&lang=zh_CN#rd)

**浦语·灵笔2-视觉问答-7B（internlm-xcomposer2-vl-7b）**

**代码库链接：**[https://github.com/InternLM/InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)

**模型链接：**[https://modelscope.cn/models/Shanghai\_AI\_Laboratory/internlm-xcomposer2-vl-7b/summary](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b/summary)

[https://modelscope.cn/models/Shanghai\_AI\_Laboratory/internlm-xcomposer2-vl-7b/summary](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b/summary)

最佳实践：[https://mp.weixin.qq.com/s?\_\_biz=MzkxNTM5NTg2OA==&mid=2247489025&idx=1&sn=7b0b3d8d6cf32a7c59d6e5421e9d874d&chksm=c15e9272f6291b6441147249afb34b52e86b05b3762c1bf677f2c98ab7cf98ca01fd3291ac68&token=740682062&lang=zh\_CN#rd](https://mp.weixin.qq.com/s?__biz=MzkxNTM5NTg2OA==&mid=2247489025&idx=1&sn=7b0b3d8d6cf32a7c59d6e5421e9d874d&chksm=c15e9272f6291b6441147249afb34b52e86b05b3762c1bf677f2c98ab7cf98ca01fd3291ac68&token=740682062&lang=zh_CN#rd)

**CogVLM模型：**

技术报告：[https://zhipu-ai.feishu.cn/wiki/LXQIwqo1OiIVTykMh9Lc3w1Fn7g](https://zhipu-ai.feishu.cn/wiki/LXQIwqo1OiIVTykMh9Lc3w1Fn7g)

代码库链接：[https://github.com/THUDM/CogVLM](https://github.com/THUDM/CogVLM)

模型链接：[https://modelscope.cn/models/ZhipuAI/CogVLM/summary](https://modelscope.cn/models/ZhipuAI/CogVLM/summary)

最佳实践：[https://mp.weixin.qq.com/s?\_\_biz=MzkxNTM5NTg2OA==&mid=2247486833&idx=1&sn=13dda47e2feca1147f7c763b7cde91dd&chksm=c15e8902f62900143a04fd2a4591887c3c4841f33c28dfa75b6da1c5d8c914edd33d767829b7&token=740682062&lang=zh\_CN#rd](https://mp.weixin.qq.com/s?__biz=MzkxNTM5NTg2OA==&mid=2247486833&idx=1&sn=13dda47e2feca1147f7c763b7cde91dd&chksm=c15e8902f62900143a04fd2a4591887c3c4841f33c28dfa75b6da1c5d8c914edd33d767829b7&token=740682062&lang=zh_CN#rd)

**MiniCPM-V模型：**

**论文链接：**[https://arxiv.org/abs/2308.12038](https://arxiv.org/abs/2308.12038)

**代码库链接：**[https://github.com/OpenBMB/OmniLMM/](https://github.com/OpenBMB/OmniLMM/)

模型链接：[https://modelscope.cn/models/OpenBMB/MiniCPM-V/summary](https://modelscope.cn/models/OpenBMB/MiniCPM-V/summary)

**Video-LLaVA模型：**

论文链接：[https://arxiv.org/abs/2311.10122](https://arxiv.org/abs/2311.10122)

代码库链接：[https://github.com/PKU-YuanGroup/Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)

模型链接：[https://modelscope.cn/models/PKU-YuanLab/Video-LLaVA-7B/summary](https://modelscope.cn/models/PKU-YuanLab/Video-LLaVA-7B/summary)

**总结对比：**

从模型参数量来看，零一万物，CogVLM的模型是百亿参数，但是仅支持英文，通义，灵笔等模型可以较好的支持中文，Video-LLaVA可以支持直接对视频的理解，可以根据需求来选择具体的多模态大语言模型。

算力评估

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/a2QnVJp637Rmn4XB/img/6af01ffa-2127-4957-9d7e-27b1e67b9ed2.png)

**可以加入的开源的opensora项目**

把最近开源的一些Sora项目简单做了整理，也分享给大家：

北大袁粒项目组：

[https://github.com/PKU-YuanGroup/Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan:)

涵盖VideoGPT做encoder/decoder 以及 实现了VideoDIT的训练

社区开发者：

[https://github.com/mini-sora/minisora](https://github.com/mini-sora/minisora：)

拆解了多条复现路线，待具体推进

NUS：

[https://github.com/NUS-HPC-AI-Lab/OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT)  

实现了VideoDit训练，注重高效训练

 Colossal-AI：

[https://github.com/hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora：)

和OpenDiT类似，实现了VideoDIT训练代码

开发者：

[https://github.com/SoraWebui/SoraWebui：](https://github.com/SoraWebui/SoraWebui：) 

从这个思路切入做开源Sora生态挺好的，只是还需要等一等，魔搭也会从工具角度出发更多的服务Sora生态

魔搭社区开发者复现进展（pixel space的初步实现，近期完成开源）：[https://github.com/modelscope/lite-sora](https://github.com/modelscope/lite-sora)
