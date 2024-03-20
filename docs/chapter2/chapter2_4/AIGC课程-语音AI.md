# AIGC课程-语音AI

# 语音AI的发展

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/d827bec4-3e14-4f08-8534-8013ff225de2.png)

本文主要关注在语音合成，以及个性化声音合成。

# 什么是TTS

TTS（Text To Speech）是一种语音AI技术，其全称为“TEXT To 文本转语音”，顾名思义，它能够将书面文字信息智能地转化为可听的、流畅自然的人类语音输出。这项技术融合了语言学、心理学、声学、数字信号处理以及神经网络等多个学科领域的知识和成果。

TTS模型将接收到的文字数据进行解析和转换。这些算法模仿发音，包括对音节、韵律、重音和语调等语音特征的精细模拟，从而生成与真人语音极为相似的声音流。不仅如此，TTS模型还能够根据上下文调整语气和语速，确保合成语音具有丰富的表现力和高自然度，从而提升了人机交互的体验。

魔搭社区上开源了丰富的（72个）语音合成模型，除了中文外，还支持英语、德语、法语、韩语等多语言模型，以及支持上海话，四川话，粤语等方言生成。

TTS应用场景包括：智能客服，朗读，视频配音等多种业务场景，同时支持多情感语音合成，可以满足各种不同类型文案的合成需求。

下面是魔搭社区开源的语音合成模型的基础框架

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/2152e306-6283-45c6-8ac4-355c5cb394c2.png)

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/3ce6949f-7073-46b0-8074-497ff50d3a5e.png)

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/65a5bd18-3e51-409a-a315-3b59cf6b83f1.png)

# 个性化语音合成

## 个性化语音合成PTTS

个性化语音生成是做什么的。我们把个性化语音生成定义为，保持相同语义，变更说话人音色。而过去的声音定制对录音的环境，时长，标注都有比较高的要求。

通义实验室智能语音实验室在魔搭社区上开源了ModelScopeTTS的模型，录制20句话即可定制模型，finetune流程只需要10分钟。

模型链接：

自动标注模型

[https://modelscope.cn/models/iic/speech\_ptts\_autolabel\_16k/summary](https://modelscope.cn/models/iic/speech_ptts_autolabel_16k/summary)

个性化语音合成模型

[https://modelscope.cn/models/iic/speech\_personal\_sambert-hifigan\_nsf\_tts\_zh-cn\_pretrain\_16k/summary](https://modelscope.cn/models/iic/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/summary)

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/9e054c78-5c62-4ed4-962c-b76773daedb4.png)

**TTS-AutoLabel**是一个集成进入**ModelScope**的语音合成自动化标注工具，旨在降低数据标注门槛，

使开发者更便捷的定制个性化语音合成模型。

模型链接：

自动标注模型

[https://modelscope.cn/models/iic/speech\_ptts\_autolabel\_16k/summary](https://modelscope.cn/models/iic/speech_ptts_autolabel_16k/summary)

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/7139c02b-c128-4052-b792-267528dd27f9.png)

0代码创空间体验链接：

[https://modelscope.cn/studios/iic/personal\_tts/summary](https://modelscope.cn/studios/iic/personal_tts/summary)

基于ModelScope NoteBook免费算力最佳实践

•Notebook最佳实践 ([https://modelscope.cn/models/damo/speech\_personal\_sambert-hifigan\_nsf\_tts\_zh-cn\_pretrain\_16k/summary](https://modelscope.cn/models/damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/summary))

点击NoteBook快速开发：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/fcdf6743-9685-4830-9c2d-c690a16e8fc2.png)

参考模型主页最佳实践：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/554383ba-e1d0-4c57-9aa9-4dab06113334.png)

## 个性化语音生成GPT-Sovits

GPT-Sovits技术来源于由语音转换领域最有名的，上了2023年时代周刊前两百大发明的算法So-Vits。So-vits的原理可以很简单的抽象为编码器解码器结构。首先通过编码器去除音色，然后通过Vits解码器重建音频。具体架构图如下：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/5c2f6b7e-f25d-42e7-9be5-c816e7ab0404.png)

so-vits-svc开源地址：[https://github.com/svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)

so-vits算法的一些特点：

VITS：文本前端+时长预测器->语义embedding->编解码后端->合成语音波形

SoVITS：波形->自监督语义embedding->编解码后端->合成语音波形

SoVITS：语义特征比文本更soft，不会强制要求输出为某个字，能更精准地表示某个时刻的音素信息

  增加音高信息强制绑定，模型不需要预测音调，因此输出音调不会很奇怪

  泄露输入源音色

  仅需音频即可训练

随着sovit+自回归模型的发展，声音相关的AIGC应用也迎来了很大的发展：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/vBPlNYk3BwezOdG8/img/06e20211-1531-4630-bf23-623ad5c082df.png)

魔搭社区开发者基于GPT-Sovits开发了各种变声应用，体验链接：

[https://modelscope.cn/studios/xzjosh/GPT-SoVITS/summary](https://modelscope.cn/studios/xzjosh/GPT-SoVITS/summary)

GPT-Sovits开源地址：

[https://github.com/RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

魔搭社区的notebook体验地址：

[https://github.com/datawhalechina/sora-tutorial/blob/main/docs/chapter2/chapter2\_4/GPT-SoVITS-demo.ipynb](https://github.com/datawhalechina/sora-tutorial/blob/main/docs/chapter2/chapter2_4/GPT-SoVITS-demo.ipynb)