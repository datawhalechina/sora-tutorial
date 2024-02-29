# AIGC技术基础知识-Stable Diffusion

# 一、AIGC是什么

首先我们来看一下什么是AIGC。

AIGC在过去的一年很火，全称叫做AI generated content，AlGC (Al-Generated Content，人工智能生产内容)，是利用AlI自动生产内容的生产方式。

在传统的内容创作领域中，PGC（Professionally-generated Content，专业生成内容）和UGC（User-generated Content，用户内容生产）作为两大主流模式，共同构成了内容生产的核心来源。然而，随着技术进步，AIGC（人工智能生成内容）的兴起正在引领一场革命，它不仅让人工智能具备了对世界的感知与理解能力，更进一步地将其延伸至创造性生成层面。这一转变预示着AIGC将在未来深刻影响并重塑各行业内容生产的范式和格局。

AIGC的发展依赖如下三个要素：

*   更强，同时也是更便宜的算力
    
*   更多的高质量数据集，包括文本、语音、视觉和多模态
    
*   模型技术的发展，更具有扩展性和更好的模型，比如Transformers和diffusion model
    

所以AIGC能做的，且做得比较好的领域越来越多，包括：

*   自然语言领域（比如代码生成、论文写作、诗歌对联、剧本创作，agent智能体）
    
*   语音领域（比如语音合成，音乐生成，个性化声音生成），
    
*   视觉领域的图像生成（stable diffusion, mid-journey）、以及最近也发展很迅速的视频生成（sora）。
    

这些都是属于AIGC的范畴，而且正快速的改变着我们的生产力工具、改变着我们的生活。本节课主要关注在视觉领域的AIGC，即图像生成和视频生成。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/d64f849f-90a5-4485-852d-f6f63294f70e.png)

# 二、AIGC技术的发展

本段介绍AIGC技术的发展，以图片生成任务为例（视频生成任务发展路径类似）

从这张图我们也可以看到，随着技术的发展，AIGC生成的图片的质量越来越高。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/1826f924-0460-47f6-9376-aafaefdbe5a0.png)

什么是文生图呢，下面这张图给了一个例子，输入文本“一只戴着太阳镜的小松鼠在演奏吉他”，经过文生图的模型，可以输出对应的一张rgb的图像。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/6449c9fc-d795-47e6-82ad-216f185a3d5c.png)

根据文生图的发展路线，我们把文生图的发展历程发展成如下4个阶段：

*   基于生成对抗网络的（GAN）模型
    
*   基于自回归(Autoregressive)模型
    
*   基于扩散(diffusion)模型
    
*   基于Transformers的扩散（diffusion）模型
    

下面我们对这四种算法模型进行简单的介绍：

## 基于生成对抗网络的（GAN）模型

生成对抗网络的基本原理可以看左侧的示意图。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/66433af2-c5dd-404f-961c-b26ade529a54.png)

2014 年，Ian J.Goodfellow 提出了 GAN，它是由一个生成器G和一个判别器D组成。生成网络产生「假」数据，并试图欺骗判别网络；训练的时候，判别网络对生成数据进行真伪鉴别，试图正确识别所有「假」数据。在训练迭代的过程中，两个网络持续地进化和对抗，直到达到平衡状态，判别网络无法再识别「假」数据。

推理的时候，只要保留生成器G就行了，输入一个随机噪声vector，生成一张图像。

右侧是一个经典的AttnGAN的框架，是一个引入了attention结构（使得图片生成局部能够和文本描述更加匹配）、并且从粗粒度到细粒度coarse to fine进行生成的框架，在当时还是取得了不错的生成效果。

GAN的优势是在一些窄分布（比如人脸）数据集上效果很好，采样速度快，方便嵌入到一些实时应用里面去。

缺点是比较难训练、不稳定，而且有Mode Collapse（模式崩塌）等问题。

## 基于自回归方式的模型

第二种方法是自回归方式，自回归方式在自然语言中用的比较多，像大家听到最多的比如GPT系列。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/ff672a18-dc07-48b4-8a2b-5c4f83ee2292.png)

VQGAN是将类似的思路拓展到了视觉生成领域。他主要包括两个步骤：

第一步：将原始的RGB图像通过vqvae或者vqgan 离散压缩成一系列的 视觉code，这些视觉code 可以利用一个训练得到的decoder恢复出原始的图像信息，当然会损失一些细节，但整体恢复质量还是OK的，特别是加了GAN loss的。

第二步：利用transformer或者GPT，来按照一定的顺序，逐个的去预测每个视觉code，当所有code都预测完了之后，就可以用第一步训练好的Decoder来生成对应的图像。因为每个code预测过程是有随机采样的，因此可以生成多样性比较高的不同图像。

这个方法比较出名的就是VQGAN，还有就是openai的dalle。

## 基于扩散（diffusion）方式的模型

扩散模型也就是我们目前大多数文生图模型所采用的技术。

扩散模型也分为两个过程，一个是前向过程，通过向原始数据不断加入高斯噪声来破坏训练数据，最终加噪声到一定步数之后，原始数据信息就完全被破坏，无限接近与一个纯噪声。另外一个过程是反向过程，通过深度网络来去噪，来学习恢复数据。

训练完成之后，我们可以通过输入随机噪声，传递给去噪过程来生成数据。这就是DDPM的基本原理。

图中是DALLE2的一个基本框架，他的整个pipeline稍微有些复杂，输入文本，经过一个多模态的CLIP模型的文本编码器，

学习一个prior网络，生成clip 图像编码，然后decoder到64\*64小图，再经过两个超分网络到256\*256，再到1024\*1024。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/16b877b4-322b-4f59-b4ef-2f95f558a03d.png)

LDM原理图：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/3594abac-cbb5-4fc8-b3f0-2410aed59b78.png)

## 基于Transformers的架构的Diffusion模型

基于Transformers的架构的Diffusion模型设计了一个简单而通用的基于Vision Transformers（ViT）的架构（U-ViT），替换了latent diffusion model中的U-Net部分中的卷积神经网络（CNN），用于diffusion模型的图像生成任务。

遵循Transformers的设计方法，这类方式将包括时间、条件和噪声图像patches在内的所有输入都视作为token。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/4b83e6d5-3492-4cb8-8590-0d2320909d9a.png)

推理链路：

第一步：输入一张256x256x3的图片,经过Encoder后得到对应的latent，压缩比为8，latent space推理时输入32x32x4的噪声，将latentspace的输入token化，图片使用patchify，label和timestep使用embedding。

第二步：结合当前的step t , 输入label y， 经过N个Dit Block通过 MLP进行输出，得到输出的噪声以及对应的协方差矩阵

第三步：经过T个step采样,得到32x32x4的降噪后的latent

在训练时，需要使得去躁后的latent和第一步得到的latent尽可能一致

# 三、使用AIGC模型以及优化AIGC生成效果

## 使用AIGC模型

魔搭社区上有丰富的文生图和文生视频模型。

文生图主要以SD系列基础模型为主，以及在其基础上微调的lora模型和人物基础模型等。

视频生成模型有ModelScope-T2V，I2VGen-XL，Animatediff等。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/1ddb01c5-c907-47de-900b-7eb23af7c13a.png)

把目前主流的Stable Diffusion模型运用起来，最重要的三个功能为提示词（prompt），参考图控制（controlnet），微调训练（lora）。

体验实战

页面体验：[https://modelscope.cn/studios/iic/scepter\_studio/summary](https://modelscope.cn/studios/iic/scepter_studio/summary)

notebook体验studio：

    pip install scepter
    python -m scepter.tools.webui --language zh

notebook体验diffusers+model：

link：待上线课程github

## 提示词

提示词很重要，一般写法：主体描述，细节描述，修饰词，艺术风格，艺术家

举个例子_【promts】Beautiful and cute girl, smiling, 16 years old, denim jacket, gradient background, soft colors, soft lighting, cinematic edge lighting, light and dark contrast, anime, super detail, 8k_

|  prompt要素  |  内容示例  |
| --- | --- |
|  主体描述  |  _Beautiful and cute girl_  |
|  细节描述  |  _16 years old, denim jacket_  |
|  动作描述  |  _smiling_  |
|  背景描述  |  gradient background  |
|  色彩描述  |  _soft colors_  |
|  灯光描述  |  _soft lighting, cinematic edge lighting,  light and dark contrast_  |
|  风格  |  _anime_  |
|  清晰度  |  _super detail, 8k_  |

_【负向prompts】(lowres, low quality, worst quality:1.2), (text:1.2), deformed, black and white,disfigured, low contrast, cropped, missing fingers_

|  负向prompt要素  |  内容示例  |
| --- | --- |
|  低质量  |  _low quality, worst quality_  |
|  色彩单一  |  _black and white_  |
|  手部问题  |  _missing fingers_  |
|  变形  |   deformed  |
|  裁剪  |  _cropped_  |
|  毁容  |  _disfigured_  |

因为文生图的prompt格式更加固定，魔搭社区有专门的咒语书，可以针对不同的场景，通过点选的方式完善prompt。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/c8f4a82b-678e-41f3-ac38-1609201b2064.png)

### 参考图控制：

ControlNet是一种用于精确控制图像生成过程的技术组件。它是一个附加到预训练的扩散模型（如Stable Diffusion模型）上的可训练神经网络模块。扩散模型通常用于从随机噪声逐渐生成图像的过程，而ControlNet的作用在于引入额外的控制信号，使得用户能够更具体地指导图像生成的各个方面（如姿势关键点、分割图、深度图、颜色等）。

下面我们举几个例子

OpenPose姿势控制：

输入是一张姿势图片（或者使用真人图片提取姿势）作为AI绘画的参考图，输入prompt后，之后AI就可以依据此生成一副相同姿势的图片；

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/36b0a411-ba92-474a-a717-25b9c6456a9d.png)  ![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/989e75e2-c27b-4063-a524-97c448d1f518.png)

Canny精准绘制

输入是一张线稿图作为AI绘画的参考图，输入prompt后，之后AI就可以根据此生成一幅根据线稿的精准绘制。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/b701040b-fba2-4bf7-9ee0-66f3fb3c3703.png)  ![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/26384cd2-99b2-4c6f-9acd-8675be2b410f.png)

Hed绘制

Hed是一种可以获取渐变线条的线稿图控制方式，相比canny更加的灵活。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/f3b4f046-34d1-4874-b6b6-80ce380d05f2.png)      ![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/ad131ae0-bfc5-453d-917f-9c416d8e99df.png)

深度图Midas

输入是一张深度图，输入prompt后，之后AI就可以根据此生成一幅根据深度图的绘制。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/ec30f15a-ca81-4cfb-8161-9bce51332831.png)   ![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/1751ad07-0802-4708-9d4b-b5f93112557a.png)

颜色color控制

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/43488459-8cb0-4efa-b1cd-ab4d7ecd0616.png)   ![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/5cd23e71-8c83-4087-8ec2-a76c4d940d53.png)

## Lora

Stable Diffusion中的Lora（LoRA）模型是一种轻量级的微调方法，它代表了“Low-Rank Adaptation”，即低秩适应。Lora不是指单一的具体模型，而是指一类通过特定微调技术应用于基础模型的扩展应用。在Stable Diffusion这一文本到图像合成模型的框架下，Lora被用来对预训练好的大模型进行针对性优化，以实现对特定主题、风格或任务的精细化控制。

prompt

Beautiful detailed illustration of the dragon in the style of Chinese spring Festival

lora选择：春节龙

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/0b019429-fcd1-4073-b434-1858d15b517c.png)

体验实战

页面体验：[https://modelscope.cn/studios/iic/scepter\_studio/summary](https://modelscope.cn/studios/iic/scepter_studio/summary)

notebook体验studio：

    pip install scepter
    python -m scepter.tools.webui --language zh

notebook体验diffusers+model：

link：待上线课程github

# 四、更多基于Stablediffusion的小应用

更多基于stablediffusion的小应用

1、facechain

[https://modelscope.cn/studios/CVstudio/cv\_human\_portrait/summary](https://modelscope.cn/studios/CVstudio/cv_human_portrait/summary)

2、InstantID

[https://modelscope.cn/studios/instantx/InstantID/summary](https://modelscope.cn/studios/instantx/InstantID/summary)

3、anytext

[https://modelscope.cn/studios/iic/studio\_anytext/summary](https://modelscope.cn/studios/iic/studio_anytext/summary)

4、replaceanything

[https://modelscope.cn/studios/iic/ReplaceAnything/summary](https://modelscope.cn/studios/iic/ReplaceAnything/summary)

5、outfitanyone

[https://modelscope.cn/studios/DAMOXR/OutfitAnyone/summary](https://modelscope.cn/studios/DAMOXR/OutfitAnyone/summary)

# 五、视频生成技术发展

过去这一年视频生成的发展路径：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/8ad8a3bf-2db2-4c5f-a8bc-e8d01c80851f.png)

基于Stable Diffusion视频生成:将视觉空间的数据映射到隐空间中，通过输入文本(或其他条件)在隐空间训练扩散模型，与图像不同的是地方在于Unet需具备时序编码的能力。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/61393618-8d1d-46a2-939b-9815b633eb45.png)

通常的视频生成的任务有两种：文生视频和图生视频

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZw2dLXgdlLX8/img/eb542d44-dfcb-4cd2-ac21-da438d1bd7f8.png)
