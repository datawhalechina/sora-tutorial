## 思考和练习

请思考下面的问题。

### Attention

1. 你怎么理解Attention？
2. 乘性Attention和加性Attention有什么不同？
3. Self-Attention为什么采用 Dot-Product Attention？
4. Self-Attention中的Scaled因子有什么作用？必须是 `sqrt(d_k)` 吗？
5. Multi-Head Self-Attention，Multi越多越好吗，为什么？
6. Multi-Head Self-Attention，固定`hidden_dim`，你认为增加 `head_dim` （需要缩小 `num_heads`）和减少 `head_dim` 会对结果有什么影响？
7. 为什么我们一般需要对 Attention weights 应用Dropout？哪些地方一般需要Dropout？Dropout在推理时是怎么执行的？你怎么理解Dropout？
8. Self-Attention的qkv初始化时，bias怎么设置，为什么？
9. 你还知道哪些变种的Attention？它们针对Vanilla实现做了哪些优化和改进？
10. 你认为Attention的缺点和不足是什么？
11. 你怎么理解Deep Learning的Deep？现在代码里只有一个Attention，多叠加几个效果会好吗？
12. DeepLearning中Deep和Wide分别有什么作用，设计模型架构时应怎么考虑？

### LLM

1. 你怎么理解Tokenize？你知道几种Tokenize方式，它们有什么区别？
2. 你觉得一个理想的Tokenizer模型应该具备哪些特点？
3. Tokenizer中有一些特殊Token，比如开始和结束标记，你觉得它们的作用是什么？我们为什么不能通过模型自动学习到开始和结束标记？
4. 为什么LLM都是Decoder-Only的？
5. RMSNorm的作用是什么，和LayerNorm有什么不同？为什么不用LayerNorm？
6. LLM中的残差连接体现在哪里？为什么用残差连接？
7. PreNormalization和PostNormalization会对模型有什么影响？为什么现在LLM都用PreNormalization？
8. FFN为什么先扩大后缩小，它们的作用分别是什么？
9. 为什么LLM需要位置编码？你了解几种位置编码方案？
10. 为什么RoPE能从众多位置编码中脱颖而出？它主要做了哪些改进？
11. 如果让你设计一种位置编码方案，你会考虑哪些因素？
12. 请你将《LLM部分》中的一些设计（如RMSNorm）加入到《Self-Attention部分》的模型设计中，看看能否提升效果？