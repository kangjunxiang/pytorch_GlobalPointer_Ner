# pytorch_GlobalPointer_Ner

Chinese Named Entity Recognition based on GlobalPointer (PyTorch).

The models come from references [1] and [2]. This project follows the existing NER template; for model details and prerequisites please refer to the links in the references. Reproduction steps:

- 1. Run `convert_data.py` to convert the raw CMeEE data into the format under `mid_data/`.
- 2. Run `main.py` with the arguments below to train, validate, test, and predict.

# Dependencies

```
pytorch==1.6.0
transformers==4.5.0
seqeval
```

# Run

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

### Results

`globalpoint2.py` is also usable. To use it, change the import in `main.py` to `import globalpoint2`, and use `globalpoint2.GlobalPointerNer` when constructing the model (set the model name to `bert-2` yourself). Note that `use_efficient_globalpointer` has no effect for `globalpoint2.py` because that flag only applies to `globalpoint.py`.

By default the model from `globalpoint.py` is used, which includes both `GlobalPointer` and `EfficientGlobalPointer`. Switch between them via the `use_efficient_globalpointer` flag.

### Notes

If the performance is poor, try reducing the learning rate.

# Acknowledgements

Thanks to the author of the TPLinker open-source project: [taishan1994](https://github.com/taishan1994)
