import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_transformer_vae_keras as tf_transformer

# Model Parameters. #
seq_encode = 50
seq_decode = 51

num_layers  = 3
num_heads   = 4
prob_keep   = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

model_ckpt_dir  = "../TF_Models/fraser_jokes_sw_transformer_vae"
train_loss_file = "train_loss_fraser_jokes_sw_transformer_vae.csv"

tmp_pkl_file = "../Data/jokes/short_jokes.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

SOS_token  = subword_2_idx["<SOS>"]
EOS_token  = subword_2_idx["<EOS>"]
PAD_token  = subword_2_idx["<PAD>"]
UNK_token  = subword_2_idx["<UNK>"]
vocab_size = len(subword_vocab)
print("Vocabulary Size:", str(vocab_size))

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Building the Transformer VAE Model.")
start_time = time.time()

seq2seq_model = tf_transformer.Transformer(
    num_layers, num_heads, hidden_size, ffwd_size, 
    vocab_size, vocab_size, seq_encode, seq_decode, 
    rate1=0.0, rate2=1.0-prob_keep, use_transformer_vae_flag=True)
seq2seq_optim = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time() - start_time) / 60
print("Transformer Model built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    seq2seq_model=seq2seq_model, 
    seq2seq_optim=seq2seq_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

train_loss_df = pd.read_csv(train_loss_file)
train_loss_list = [tuple(
    train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]

# Placeholders to store the batch data. #
n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Inference of the Transformer VAE Network", 
      "(" + str(n_iter), "iterations).")

# Select 2 inputs to interpolate between. #
tmp_input_1 = input("Enter 1st input: ")
tmp_input_2 = input("Enter 2nd input: ")
print("-" * 50)

tmp_sw_toks_1 = bpe.bp_encode(
    tmp_input_1, subword_vocab, subword_2_idx)
tmp_sw_toks_2 = bpe.bp_encode(
    tmp_input_2, subword_vocab, subword_2_idx)
n_sw_tokens_1 = len(tmp_sw_toks_1)
n_sw_tokens_2 = len(tmp_sw_toks_2)

tmp_array_1 = np.zeros([1, seq_encode], dtype=np.int32)
tmp_array_2 = np.zeros([1, seq_encode], dtype=np.int32)

tmp_array_1[:, :] = PAD_token
tmp_array_2[:, :] = PAD_token
tmp_array_1[0, :n_sw_tokens_1] = tmp_sw_toks_1
tmp_array_2[0, :n_sw_tokens_2] = tmp_sw_toks_2

# Get the encoder latent space embedding. #
tmp_embed_1 = seq2seq_model.encoder_model(
    tmp_array_1, training=False)
tmp_embed_2 = seq2seq_model.encoder_model(
    tmp_array_2, training=False)

tmp_embed_1 = tf.expand_dims(
    tf.reduce_mean(tmp_embed_1, axis=1), axis=1)
tmp_embed_2 = tf.expand_dims(
    tf.reduce_mean(tmp_embed_2, axis=1), axis=1)

z_condition_1 = seq2seq_model.z_project(tmp_embed_1)
z_condition_2 = seq2seq_model.z_project(tmp_embed_2)

tmp_mean_1 = z_condition_1[:, :, :hidden_size]
tmp_mean_2 = z_condition_2[:, :, :hidden_size]
tmp_std_1  = tf.exp(
    0.5 * z_condition_1[:, :, hidden_size:])
tmp_std_2  = tf.exp(
    0.5 * z_condition_2[:, :, hidden_size:])

tmp_std_step  = (tmp_std_2 - tmp_std_1) / 10.0
tmp_mean_step = (tmp_mean_2 - tmp_mean_1) / 10.0

x_SOS_tok = tf.tile([SOS_token], [1])
for n in range(11):
    tmp_std  = tmp_std_1 + tmp_std_step * n
    tmp_mean = tmp_mean_1 + tmp_mean_step * n
    
    gen_ids = [tf.expand_dims(x_SOS_tok, axis=1)]
    for step in range(seq_decode):
        tmp_decode = tf.concat(gen_ids, axis=1)
        tmp_tuple  = seq2seq_model.decoder_model(
            tmp_decode, tmp_mean, tmp_std, training=False)
        
        tmp_logits = seq2seq_model.p_decoder(tmp_tuple[1])
        tmp_index  = tf.argmax(
            tmp_logits[:, -1, :], 
            axis=-1, output_type=tf.int32)
        gen_ids.append(tf.expand_dims(tmp_index, axis=1))
    
    gen_ids = tf.concat(gen_ids, axis=1)
    gen_phrase = bpe.bp_decode(
        gen_ids.numpy()[0], idx_2_subword)
    
    gen_decoded_phrase = " ".join(
        gen_phrase).replace("<", "").replace(">", "")
    
    print("Generated Phrase:")
    print(gen_decoded_phrase)
    print("-" * 50)
