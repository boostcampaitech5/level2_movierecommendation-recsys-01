import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    """
    A single encoder layer for BERT4Rec.
    
    Arguments
    ----------
    embed_dim (int): The dimension of the embedding vectors.
    n_heads (int): The number of heads in the multihead attention.
    pffn_hidden_dim (int): The dimension of the hidden layer in the pointwise feed forward network.
    unidirection (bool): A boolean value to specify the direction of the attention.
    dropout_rate (float): The dropout rate.
    """
    def __init__(self,
                 embed_dim: int,
                 n_heads: int,
                 pffn_hidden_dim: int,
                 unidirection: bool,
                 dropout_rate: float):
        super(EncoderLayer, self).__init__()        
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, dropout=dropout_rate)
        self.pffn = nn.Sequential(
                    nn.Linear(embed_dim, pffn_hidden_dim),
                    nn.GELU(),
                    nn.Linear(pffn_hidden_dim, embed_dim)
        )
        self.unidirection = unidirection
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, embed_seq, padding_mask):
        """
        Forward pass of the EncoderLayer.
        attention -> residual connection -> pointwise feed forward network -> residual connection

        Arguments
        ----------
        embed_seq (torch.Tensor): The input sequence tensor. Shape (batch_size, max_len, embed_dim).
        padding_mask (torch.Tensor): A mask to indicate zero-padding. Shape (batch_size, max_len).

        Returns
        ----------
        out (torch.Tensor): The output sequence. Shape (batch_size, max_len, embed_dim).
        """
        embed_seq = embed_seq.transpose(0, 1)
        mha_out, _ = self.mha(embed_seq, embed_seq, embed_seq,
                              key_padding_mask=padding_mask,
                              is_causal=self.unidirection)
        mha_out = mha_out.transpose(0, 1)
        mha_out = self.layer_norm1(self.dropout(mha_out) + embed_seq.transpose(0, 1))
        
        pffn_out = self.pffn(mha_out)
        out = self.layer_norm2(self.dropout(pffn_out) + mha_out)
        
        return out
    
    
class BERT4Rec(nn.Module):
    """
    The BERT4Rec model, a transformer-based model for sequential recommendation.

    Arguments
    ----------
    n_items (int): The total number of uniuqe items.
    embed_dim (int): The dimension of the embedding vectors.
    max_len (int): The maximum length of the sequences.
    n_layers (int): The number of encoder layers.
    n_heads (int): The number of heads in the multihead attention.
    pffn_hidden_dim (int): The dimension of the hidden layer in the pointwise feed forward network.
    unidirection (bool): A boolean value to specify the direction of the attention.
    dropout_rate (float): The dropout rate.
    device (torch.device): The device where the model will be allocated.
    """
    def __init__(self,
                 n_items: int,
                 embed_dim: int,
                 max_len: int,
                 n_layers: int,
                 n_heads: int,
                 pffn_hidden_dim: int,
                 unidirection: bool,
                 dropout_rate: float,
                 device: torch.device):
        super(BERT4Rec, self).__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.n_layers = n_layers
        self.unidirection = unidirection
        self.device = device
        
        self.item_embed = nn.Embedding(n_items+2, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        self.encoder_layer = nn.ModuleList(
            [EncoderLayer(embed_dim, n_heads, pffn_hidden_dim, unidirection, dropout_rate)
             for _ in range(self.n_layers)]
            )
        
        self.out_layer = nn.Linear(embed_dim, n_items+1)
        
        
    def embedding_layer(self, seq: torch.tensor) -> torch.tensor:
        """
        Combines item and position embeddings into a single embedding.

        Arguments
        ----------
        seq (torch.Tensor): The input sequence. Shape (batch_size, max_len).

        Returns:
        embed_seq (torch.Tensor): The combined embedding sequence. Shape (batch_size, max_len, embed_dim).
        """
        item_embed = self.item_embed(seq)
        pos = torch.arange(self.max_len, device=self.device).unsqueeze(0)
        pos_embed = self.pos_embed(pos).repeat(item_embed.size(0), 1, 1)
        embed_seq = item_embed + pos_embed
        
        return embed_seq
    
    def forward(self, seq: torch.tensor) -> torch.tensor:
        """
        Forward pass of the BERT4Rec.
        embedding -> encoder -> output

        Arguments
        ----------
        seq (torch.Tensor): The input sequence. Shape (batch_size, max_len).

        Returns
        ----------
        out (torch.Tensor): The output sequence. Shape (batch_size, max_len, n_items+1).
        """
        embed_seq = self.embedding_layer(seq)
        
        padding_mask = (seq == 0).bool().to(self.device)
        out = embed_seq
        for block in self.encoder_layer:
            out = block(out, padding_mask)
            
        out = self.out_layer(out)
        
        return out