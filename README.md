# pytorch_GlobalPointer_Ner

基于pytorch的GlobalPointer进行中文命名实体识别。

模型分别来自于参考中的【1】【2】。这里还是按照之前命名实体识别的相关模板，具体模型的介绍及预备知识请移步参考里面的链接。复现方式：

- 1、使用 convert_data.py 将CMeEE 原始数据处理为mid_data下的数据。
- 2、根据参数运行main.py以进行训练、验证、测试和预测。

# 依赖

```
pytorch==1.6.0
transformers==4.5.0
seqeval
```

# 运行

```python
python main.py \
--bert_dir="model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/CMeEE/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=9 \
--head_size=64 \
--seed=42 \
--gpu_ids="0" \
--max_seq_len=512 \
--lr=5e-5 \
--other_lr=5e-5 \
--train_batch_size=42 \
--train_epochs=7 \
--eval_steps=50 \
--eval_batch_size=8 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--dropout_prob=0.1 \
--use_tensorboard="True" \
--use_efficient_globalpointer="True"
```

### 结果

globalpoint2.py也是可以用的，要选择它需要将main.py导入修改为```import globalpoint2```，并在使用模型时改为```globalpoint2.GlobalPointerNer```，模型名字自己设置为bert-2，参数use_efficient_globalpointer没有作用，因为是针对globalpoint.py的。


默认使用的是globalpoint.py里面的模型，包含globalpointer和efficient-globalpoint，通过修改use_efficient_globalpointer来指定选择的模型


### 补充

如果效果不好，尝试调小一些学习率。

# 鸣谢

感谢TPLinker开源项目作者 [taishan1994](https://github.com/taishan1994)