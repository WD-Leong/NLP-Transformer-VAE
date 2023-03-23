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
print("Transformer VAE Model built", 
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
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)

# Extract the number of updates done. #
n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Training the Transformer VAE Keras Network", 
      "(" + str(n_iter), "iterations).")
print("-" * 50)

while True:
    tmp_phrase = input("Enter prompt: ")
    tmp_phrase = tmp_phrase.lower().strip()

    if tmp_phrase == "":
        break
    else:
        i_encode = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword_2_idx)
        n_input  = len(i_encode)
        i_decode = bpe.bp_decode(i_encode, idx_2_subword)
        
        tmp_test_in[:, :] = PAD_token
        tmp_test_in[0, :n_input] = i_encode
        
        gen_ids = seq2seq_model.infer(
            tmp_test_in, SOS_token)
        gen_phrase = bpe.bp_decode(
            gen_ids.numpy()[0], idx_2_subword)
        
        print("Input Phrase:")
        print(" ".join(i_decode).replace("<", "").replace(">", ""))
        print("Generated Phrase:")
        print(" ".join(gen_phrase).replace("<", "").replace(">", ""))
        print("-" * 50)
