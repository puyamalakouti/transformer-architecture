import torch
import torch.nn as nn

device = 'cude' if torch.cuda.is_available() else 'cpu'

num_heads = 8
embed_size = 512
num_layers = 6
p_dropout = 0.1
batch_size = 8

input_vocab_size = 7000
output_vocab_size = 7000

class InputEmbedding(nn.Module):

    def __init__(self, input_vocab_size=input_vocab_size, embed_size=embed_size, p_dropout=p_dropout):
        super(InputEmbedding, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.embed_size = embed_size
        self.p_dropout = p_dropout

        self.word_embedding = nn.Embedding(self.input_vocab_size, self.embed_size)
        self.position_embeding = nn.Embedding(self.input_vocab_size, self.embed_size)
        self.dropout_layer = nn.Dropout(p_dropout)

    def forward(self, input):
        word_embd = self.word_embedding(input)
        batch_size, seq_len = input.shape

        position_vec = torch.arange(0, seq_len).expand(batch_size, seq_len)
        positional_enc = self.position_embeding(position_vec)
        return self.dropout_layer(word_embd + positional_enc)
    
class ScaledDotProduct(nn.Module):
    def __init__(self, embed_size=embed_size, mask=None):
        super(ScaledDotProduct, self).__init__()
        self.embed_size = embed_size
        self.mask = mask
        self.scale = self.embed_size ** 0.5

    def forward(self, query, key, value):

        # input_shape = (batch_size, num_heads, seq_len, head_length)
        compatability = torch.einsum('bhqf,bhkf->bhqk', [query, key])
        # output_shape: (batch_size, num_heads, seq_len, seq_len)

        # apply mask layer
        if self.mask is not None:
            compatability = torch.tril(compatability)

        weight = torch.softmax(compatability / self.scale, dim=3)
        # output_shape: (batch_size, num_heads, seq_len, seq_len)

        # input_shape: (batch_size, seq_len, num_heads, embed_size)
        out = torch.einsum("nhql,nlhd->nqhd", [weight, value])
        # output_shape = (batch_size, seq_len, num_heads, head_length)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size=embed_size, num_heads=num_heads, batch_size=batch_size, mask=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mask = mask
        self.head_len = self.embed_size // self.num_heads

        assert (
            self.head_len * self.num_heads == self.embed_size
        ), "The embedding size should be evenly divisible by the number of heads"

        self.fc_V = nn.Linear(self.head_len, self.head_len, bias=False)
        self.fc_K = nn.Linear(self.head_len, self.head_len, bias=False)
        self.fc_Q = nn.Linear(self.head_len, self.head_len, bias=False)

        if self.mask is not None:
            self.attention = ScaledDotProduct(mask=True)
        else:
            self.attention = ScaledDotProduct()

        self.fc_output = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, query, key, value):

        # batch_size
        n_batch = query.shape[0]

        # sequence length
        query_len, value_len, key_len = query.shape[1], value.shape[1], key.shape[1]

        # reshape query to (batch_size, num_heads, seq_len, head_length)
        query = query.reshape(n_batch, self.num_heads, query_len, self.head_len)

        # reshape key to (batch_size, num_heads, seq_len, head_length)
        key = key.reshape(n_batch, self.num_heads, key_len, self.head_len)

        # reshape value to (batch_size, seq_len, num_heads, head_length)
        value = value.reshape(n_batch, value_len, self.num_heads, self.head_len)

        query = self.fc_Q(query)
        key = self.fc_K(key)
        value = self.fc_V(value)

        sdp_output = self.attention.forward(query, key, value).reshape(
            self.batch_size, query_len, self.num_heads*self.head_len
        )

        return self.fc_output(sdp_output)
    
class Encoder(nn.Module):

    def __init__(self, embed_size=embed_size, p_dropout=p_dropout):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.p_dropout = p_dropout

        self.mha = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.dropout_layer = nn.Dropout(self.p_dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, 4*self.embed_size),
            nn.ReLU(),
            nn.Linear(4*self.embed_size, self.embed_size)
        )


    def forward(self, query, key, value):
        attention = self.mha.forward(query, key, value)
        first_output = self.dropout_layer(self.norm1(attention + query))
        forward = self.feed_forward(first_output)
        out = self.dropout_layer(self.norm2(forward + first_output))
        return out
    
class Decoder(nn.Module):

    def __init__(self, embed_size=embed_size, p_dropout=p_dropout):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.p_dropout = p_dropout

        self.masked_mha = MultiHeadAttention(mask=True)
        self.norm = nn.LayerNorm(self.embed_size)
        self.dropout_layer = nn.Dropout(self.p_dropout)
        self.encoder_block = Encoder()



    def forward(self, query, key, value):
        masked_attention = self.masked_mha.forward(query, query, query)
        query = self.dropout_layer(self.norm(masked_attention + query))
        out = self.encoder_block(query, key, value)
        return out
    
class Transformer(nn.Module):

    def __init__(self, embed_size=embed_size, num_layers=num_layers, output_vocab_size=output_vocab_size, device=device):
        super(Transformer, self).__init__()
        self.embed_len = embed_size
        self.num_layers = num_layers
        self.output_vocab_size = output_vocab_size
        self.device = device

        self.embedding = InputEmbedding().to(self.device)

        self.encoder_block = nn.ModuleList(
            [Encoder() for _ in range(self.num_layers)]
        )

        self.decoder_block = nn.ModuleList(
            [Decoder() for _ in range(self.num_layers)]
        )

        self.output_layer = nn.Sequential(
            nn.Linear(self.embed_len, self.output_vocab_size).to(self.device),
            nn.Softmax(dim=2)
        )

    def forward(self, source, target):
        enc_output = self.embedding(source)

        for layer in self.encoder_block:
            enc_output = layer(enc_output, enc_output, enc_output)

        dec_output = self.embedding(target)
        for layer in self.decoder_block:
            dec_output = layer(dec_output, enc_output, enc_output)

        final_output = self.output_layer(dec_output)
        return final_output

if __name__ == "__main__":

    src = torch.randint(10, (batch_size, 30))
    trg = torch.randint(10, (batch_size, 20))

    model = Transformer()

    out = model.forward(src, trg)
    print(out.shape)