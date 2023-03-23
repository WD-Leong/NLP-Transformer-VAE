# Import the libraries. #
import tensorflow as tf
from tensorflow.keras.layers import (
    LayerNormalization, Embedding)

# Multi-Head Attention Layer. #
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = int(d_model / n_heads)
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wc = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x):
        # Input is (batch_size, seq_len, d_model). #
        # Output is (batch_size, num_heads, seq_len, depth). #
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        output_shp = (batch_size, seq_length, 
                      self.n_heads, self.d_depth)
        
        x = tf.reshape(x, output_shp)
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        output_shp = (
            batch_size, seq_length, self.d_model)
        
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, output_shp)
    
    def scaled_dot_product_attention(
        self, q, k, v, mask=None, neg_infty=-1.0e9):
        # Head dimension. #
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        lq = tf.shape(q)[2]
        lk = tf.shape(k)[2]
        
        # Multiplicative Attention. #
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale multiplicative attention mechanism. #
        attn_logits = matmul_qk * tf.math.rsqrt(dk)
        
        # Add the mask to the attention mechanism. #
        if mask is not None:
            attn_mask = (mask * neg_infty)
        else:
            attn_mask = tf.zeros([lq, lk])
        attn_logits += attn_mask
        
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_outputs = tf.matmul(attn_weights, v)
        return attn_outputs, attn_weights

    def call(self, q, k, v, mask=None):
        q_input = self.split_heads(self.wq(q))
        k_input = self.split_heads(self.wk(k))
        v_input = self.split_heads(self.wv(v))
        
        attn_tuple = self.scaled_dot_product_attention(
            q_input, k_input, v_input, mask=mask)
        
        attn_wgt = attn_tuple[1]
        attn_out = self.combine_heads(attn_tuple[0])
        attn_out = self.wc(attn_out)
        return attn_out, attn_wgt
        
class FFWNetwork(tf.keras.layers.Layer):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = tf.keras.layers.Dense(
            d_ffwd, activation="relu")
        self.ffwd_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        return self.ffwd_2(self.ffwd_1(x))

# Transformer Encoder Layer. #
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_ffwd, rate1=0.1, rate2=0.1):
        super(EncoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(d_model, n_heads)
        
        self.lnorm_1 = LayerNormalization(epsilon=1.0e-6)
        self.lnorm_2 = LayerNormalization(epsilon=1.0e-6)
        
        self.dropout_1 = tf.keras.layers.Dropout(rate1)
        self.dropout_2 = tf.keras.layers.Dropout(rate2)
    
    def call(self, x_enc, x_pos, training=True):
        x_embed = x_enc + x_pos
        
        attn_self_tuple = self.attn_self(
            x_embed, x_embed, x_embed, mask=None)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = self.dropout_1(
            attn_self_tuple[0], training=training)
        attn_self_output = tf.add(
            x_embed, self.lnorm_1(attn_self_output))
        
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = tf.add(
            attn_self_output, ffwd_self_output)
        ffwd_self_output = self.dropout_2(
            ffwd_self_output, training=training)
        return ffwd_self_output

class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, d_model, n_heads, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(Encoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.vocab_size = vocab_size
        
        # Embedding layers. #
        tmp_pos_embed = []
        for n_layer in range(n_layers):
            tmp_pos_embed.append(
                Embedding(max_seq_length, d_model))
        
        self.pos_embed = tmp_pos_embed
        self.enc_embed = Embedding(vocab_size, d_model)
        del tmp_pos_embed
        
        # Encoder Layers. #
        tmp_enc_layers = []
        for n_layer in range(n_layers):
            tmp_enc_layers.append(EncoderLayer(
                d_model, n_heads, d_ffwd, rate1, rate2))
        
        self.enc_layers = tmp_enc_layers
        self.emb_dropout = tf.keras.layers.Dropout(rate1)
        del tmp_enc_layers
    
    def call(self, x, training=True):
        seq_length = tf.shape(x)[1]
        
        x_pos_index = tf.expand_dims(
            tf.range(seq_length), axis=0)
        x_tok_embed = self.enc_embed(x)
        x_tok_embed = self.emb_dropout(
            x_tok_embed * self.d_rsqrt, training=training)
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = self.emb_dropout(
                x_pos_embed * self.d_rsqrt, training=training)
            
            layer_output = self.enc_layers[m](
                layer_input, x_pos_embed, training=training)
            layer_input  = layer_output
        return layer_output

# Transformer Decoder Layer. #
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_ffwd, rate1=0.1, rate2=0.1):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(d_model, n_heads)

        self.ffwd_enc_dec = FFWNetwork(d_ffwd, d_model)
        self.attn_enc_dec = MultiHeadAttention(d_model, n_heads)
        
        self.lnorm_1 = LayerNormalization(epsilon=1.0e-6)
        self.lnorm_2 = LayerNormalization(epsilon=1.0e-6)
        self.lnorm_3 = LayerNormalization(epsilon=1.0e-6)
        self.lnorm_4 = LayerNormalization(epsilon=1.0e-6)
        
        self.dropout_1 = tf.keras.layers.Dropout(rate1)
        self.dropout_2 = tf.keras.layers.Dropout(rate2)
        self.dropout_3 = tf.keras.layers.Dropout(rate1)
        self.dropout_4 = tf.keras.layers.Dropout(rate2)
    
    def call(
        self, x_dec, x_pos, 
        mask, z_sampled, training=True):
        x_embed = x_dec + x_pos
        
        # Decoder Self-Attention. #
        attn_self_tuple = self.attn_self(
            x_embed, x_embed, x_embed, mask=mask)
        
        # Apply Normalisation followed by adding. #
        attn_self_output = self.dropout_1(
            attn_self_tuple[0], training=training)
        attn_self_output = tf.add(
            x_embed, self.lnorm_1(attn_self_output))
        
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = tf.add(
            attn_self_output, ffwd_self_output)
        ffwd_self_output = self.dropout_2(
            ffwd_self_output, training=training)
        
        # Encoder-Decoder Attention. #
        attn_enc_dec_tuple = self.attn_enc_dec(
            ffwd_self_output, z_sampled, z_sampled)
        
        # Apply Normalisation followed by adding. #
        attn_enc_dec_output = self.dropout_3(
            attn_enc_dec_tuple[0], training=training)
        attn_enc_dec_output = tf.add(
            ffwd_self_output, self.lnorm_3(attn_enc_dec_output))
        
        ffwd_enc_dec_output = self.lnorm_4(
            self.ffwd_enc_dec(attn_enc_dec_output))
        ffwd_enc_dec_output = tf.add(
            attn_enc_dec_output, ffwd_enc_dec_output)
        ffwd_enc_dec_output = self.dropout_4(
            ffwd_enc_dec_output, training=training)
        return ffwd_enc_dec_output

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, d_model, n_heads, d_ffwd, 
        vocab_size, max_seq_length, rate1=0.1, rate2=0.1):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.vocab_size = vocab_size
        
        # Embedding layers. #
        tmp_pos_embed = []
        for n_layer in range(n_layers):
            tmp_pos_embed.append(
                Embedding(max_seq_length, d_model))
        
        self.pos_embed = tmp_pos_embed
        self.dec_embed = Embedding(vocab_size, d_model)
        del tmp_pos_embed
        
        # Decoder Layers. #
        tmp_dec_layers = []
        for n_layer in range(n_layers):
            tmp_dec_layers.append(DecoderLayer(
                d_model, n_heads, d_ffwd, rate1, rate2))
        
        self.dec_layers = tmp_dec_layers
        self.emb_dropout = tf.keras.layers.Dropout(rate1)
        del tmp_dec_layers
    
    def call(self, x, z_mean, z_std, training=True):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        enc_length = tf.shape(z_mean)[1]
        input_mask = tf.linalg.band_part(
            tf.ones([seq_length, seq_length]), -1, 0)
        input_mask = 1.0 - input_mask
        
        x_pos_index = tf.expand_dims(
            tf.range(seq_length), axis=0)
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = self.emb_dropout(
            x_tok_embed * self.d_rsqrt, training=training)
        
        x_epsilon = tf.add(
            z_mean, tf.multiply(z_std, tf.random.normal(
                shape=(batch_size, enc_length, self.d_model))))
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = self.emb_dropout(
                x_pos_embed * self.d_rsqrt, training=training)
            
            layer_output = self.dec_layers[m](
                layer_input, x_pos_embed, 
                input_mask, x_epsilon, training=training)
            layer_input  = layer_output
        return x_epsilon, layer_output

class Transformer(tf.keras.Model):
    def __init__(
        self, n_layers, n_heads, d_model, d_ffwd, 
        enc_vocab_size, dec_vocab_size, enc_len, dec_len, 
        rate1=0.1, rate2=0.1, use_transformer_vae_flag=False):
        super(Transformer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.enc_seq_len = enc_len
        self.dec_seq_len = dec_len
        self.use_vae_flag = use_transformer_vae_flag
        
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size

        # Transformer Encoder. #
        self.encoder_model = Encoder(
            n_layers, d_model, n_heads, 
            d_ffwd, enc_vocab_size, 
            enc_len, rate1=rate1, rate2=rate2)
        
        # Transformer Decoder. #
        self.decoder_model = Decoder(
            n_layers, d_model, n_heads, 
            d_ffwd, dec_vocab_size, 
            dec_len, rate1=rate1, rate2=rate2)
        
        # For VAE Mean and Variance, and output projection. #
        if use_transformer_vae_flag:
            # Setting use_transformer_vae_flag to False reverts the model #
            # to the standard Transformer Sequence-to-Sequence model.     #
            self.z_project = tf.keras.layers.Dense(2*d_model)
        self.p_decoder = tf.keras.layers.Dense(dec_vocab_size)
    
    def call(self, x_encode, x_decode, training=True):
        enc_outputs = self.encoder_model(
            x_encode, training=training)
        
        # Take average encoder outputs as the conditioning #
        # for the decoder output.                          #
        if self.use_vae_flag:
            encoder_mean = tf.expand_dims(
                tf.reduce_mean(enc_outputs, axis=1), axis=1)
            z_condition  = self.z_project(encoder_mean)

            z_std  = tf.exp(
                0.5 * z_condition[:, :, self.d_model:])
            z_mean = z_condition[:, :, :self.d_model]
        else:
            z_std  = 0.0
            z_mean = enc_outputs
        
        dec_tuple  = self.decoder_model(
            x_decode, z_mean, z_std, training=training)
        z_sampled  = dec_tuple[0]
        dec_logits = self.p_decoder(dec_tuple[1])
        return z_sampled, z_mean, z_std, dec_logits
    
    def infer(self, x_encode, SOS_token):
        batch_size = tf.shape(x_encode)[0]
        
        x_SOS_tok = tf.tile([SOS_token], [batch_size])
        infer_ids = [tf.expand_dims(x_SOS_tok, axis=1)]
        
        enc_outputs = self.encoder_model(
            x_encode, training=False)
        
        if self.use_vae_flag:
            encoder_mean = tf.expand_dims(
                tf.reduce_mean(enc_outputs, axis=1), axis=1)
            z_condition  = self.z_project(encoder_mean)

            z_std  = tf.exp(
                0.5 * z_condition[:, :, self.d_model:])
            z_mean = z_condition[:, :, :self.d_model]
        else:
            z_std  = 0.0
            z_mean = enc_outputs
        
        for step in range(self.dec_seq_len):
            tmp_decode = tf.concat(infer_ids, axis=1)
            tmp_tuple  = self.decoder_model(
                tmp_decode, z_mean, z_std, training=False)
            
            tmp_logits = self.p_decoder(tmp_tuple[1])
            tmp_index  = tf.argmax(
                tmp_logits[:, -1, :], 
                axis=-1, output_type=tf.int32)
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
