```bash
conda init
conda create -n seq python=3.9 -y
conda activate seq
pip install -r requirements.txt
```


- CONFIG
  - data_dir: Directory of data. default="../../data/"
  - model_dir: Directory of model saved. default="./model/"
  - output_dir: Directory of output saved. default="./output/"
  - seed: Random seed. default=100
  - max_len: Maximum length of a sequence. default=100
  - embed_dim: Embedding dimension. default=128
  - n_layers: Number of encoder layers. default=2
  - n_heads: Number of heads for attention. default=4
  - pffn_hidden_dim: Hidden dimension of point-wise feed forward network in encoder layer. default=512
  - dropout_rate: Dropout rate. default=0.5
  - mask_prob: Masking probabilty. default=0.15
  - batch_size: Batch size. default=128
  - lr: Learning rate. default=0.001
  - n_epochs: Number of epochs. default=200
  - max_patience: Maximum patience count. default=20
  - n_neg_samples: Number of negative samples for validation. default=1000
  - k: Number of recommendation. default=10