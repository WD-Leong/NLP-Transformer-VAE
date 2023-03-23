import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_transformer_vae_keras as tf_transformer

# Define the log Normal PDF. #
def log_normal_pdf(sample, mean, var):
    log2pi = tf.math.log(2.0 * np.pi)
    logvar = tf.math.log(var)
    
    sq_diff  = tf.square(sample - mean)
    log_vals = tf.add(
        logvar + log2pi, tf.divide(sq_diff, var))
    pdf_eval = tf.reduce_mean(
        -0.5 * tf.reduce_mean(log_vals, axis=2), axis=1)
    
    # Reduce sum should be used but this will dominate the loss and #
    # prevent the decoder from learning the training examples, so   #
    # reduce_mean is applied to reduce the impact on p(x|z).        #
    #pdf_eval = tf.reduce_sum(
    #    -0.5 * tf.reduce_sum(log_vals, axis=2), axis=1)
    return pdf_eval

# Define the weight update step. #
def train_step(
    model, sub_batch_sz, 
    x_encode, x_decode, x_output, optimizer, 
    learning_rate=1.0e-3, gradient_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        n_sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        n_sub_batch = int(batch_size / sub_batch_sz)
    else:
        n_sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]

    ce_losses  = 0.0
    pz_losses  = 0.0
    qzx_losses = 0.0
    tot_losses = 0.0
    for n_sub in range(n_sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (n_sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = x_encode[id_st:id_en, :]
        tmp_decode = x_decode[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en, :]
        with tf.GradientTape() as grad_tape:
            output_tuple = model(tmp_encode, tmp_decode)
            z_sampled = output_tuple[0]
            z_mean = output_tuple[1]
            z_var  = tf.square(output_tuple[2])

            logpz = log_normal_pdf(
                z_sampled, 0.0, 1.0)
            logqz_x = log_normal_pdf(
                z_sampled, z_mean, z_var)
            logpx_z = -1.0 * tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=output_tuple[3]), axis=1)
            
            # Compute the loss objective. #
            if model.use_vae_flag:
                # Total VAE Loss (based on ELBO). #
                tmp_losses = -1.0 * tf.reduce_sum(
                    logpx_z + logpz - logqz_x)
            else:
                # Cross-Entropy Loss for standard Transformer. #
                tmp_losses = -1.0 * tf.reduce_sum(logpx_z)
        
        pz_losses  += -1.0 * tf.reduce_sum(logpz)
        qzx_losses += -1.0 * tf.reduce_sum(logqz_x)
        ce_losses  += -1.0 * tf.reduce_sum(logpx_z)
        
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    avg_ce_loss  = ce_losses / batch_size
    avg_pz_loss  = pz_losses / batch_size
    avg_qzx_loss = qzx_losses / batch_size
    avg_tot_loss = tot_losses / batch_size

    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, gradient_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return avg_tot_loss, avg_ce_loss, avg_pz_loss, avg_qzx_loss

# Model Parameters. #
sub_batch_sz = 128
batch_size = 256
seq_encode = 50
seq_decode = 51

num_layers  = 3
num_heads   = 4
prob_keep   = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

initial_lr    = 0.001
gradient_clip = 1.00
maximum_iter  = 20000
restore_flag  = True
display_step  = 50
cooling_step  = 100
warmup_steps  = 5000
anneal_step   = 2000
anneal_rate   = 0.75

model_ckpt_dir  = "../TF_Models/fraser_jokes_sw_transformer_vae"
train_loss_file = "train_loss_fraser_jokes_sw_transformer_vae.csv"

tmp_pkl_file = "../Data/jokes/short_jokes.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    subword_vocab = pkl.load(tmp_load_file)
    idx_2_subword = pkl.load(tmp_load_file)
    subword_2_idx = pkl.load(tmp_load_file)

# Filter the dataset. #
filtered_data = []
for tmp_data in full_data:
    if len(tmp_data) <= seq_encode \
        and len(tmp_data) <= (seq_decode-1):
        filtered_data.append(tmp_data)
del tmp_data, full_data

jokes_data = filtered_data
vocab_size = len(subword_vocab)
print("Vocabulary Size:", str(vocab_size))
del filtered_data

num_data  = len(jokes_data)
SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

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

if restore_flag:
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
else:
    print("Training a new model.")
    train_loss_list = []

# Placeholders to store the batch data. #
tmp_input   = np.zeros(
    [batch_size, seq_encode], dtype=np.int32)
tmp_seq_out = np.zeros(
    [batch_size, seq_decode+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)

# Extract the number of updates done. #
n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Training the Transformer VAE Keras Network", 
      "(" + str(n_iter), "iterations).")
print("Total of", str(num_data), "training samples.")

ce_loss  = 0.0
pz_loss  = 0.0
qzx_loss = 0.0
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_input[:, :]   = PAD_token
    tmp_seq_out[:, :] = PAD_token
    tmp_seq_out[:, 0] = SOS_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_idx = jokes_data[tmp_index]
        tmp_o_idx = jokes_data[tmp_index] + [EOS_token]

        # Randomly shuffle the input data. #
        tmp_idx  = np.random.permutation(len(tmp_i_idx))
        n_sample = np.random.randint(1, len(tmp_i_idx))
        i_sample = [tmp_i_idx[x] for x in tmp_idx[:n_sample]]

        n_input  = len(i_sample)
        n_output = len(tmp_o_idx)

        tmp_input[n_index, :n_input] = i_sample
        tmp_seq_out[n_index, :n_output] = tmp_o_idx

    tmp_decode = tmp_seq_out[:, :-1]
    tmp_output = tmp_seq_out[:, 1:]
    
    tmp_loss = train_step(
        seq2seq_model, sub_batch_sz, 
        tmp_input, tmp_decode, tmp_output, 
        seq2seq_optim, learning_rate=learn_rate_val)
    
    n_iter += 1
    ce_loss  += tmp_loss[1].numpy()
    pz_loss  += tmp_loss[2].numpy()
    qzx_loss += tmp_loss[3].numpy()
    tot_loss += tmp_loss[0].numpy()
    ckpt.step.assign_add(1)

    if n_iter % display_step == 0:
        end_time = time.time()
        avg_pz  = pz_loss / display_step
        avg_qzx = qzx_loss / display_step

        avg_xent = ce_loss / display_step
        avg_loss = tot_loss / display_step

        ce_loss  = 0.0
        pz_loss  = 0.0
        qzx_loss = 0.0
        tot_loss = 0.0

        elapsed_tm = (end_time - start_tm) / 60
        start_tm   = time.time()

        tmp_test_in[:, :] = PAD_token
        sample_id = np.random.choice(num_data)
        tmp_i_idx = jokes_data[sample_id]
        tmp_o_tok = bpe.bp_decode(tmp_i_idx, idx_2_subword)

        test_input = " ".join(
            tmp_o_tok).replace("<", "").replace(">", "")
        test_token = test_input.split(" ")

        # Randomly shuffle the input data. #
        tmp_idx  = np.random.permutation(len(test_token))
        n_sample = np.random.randint(1, len(test_token))
        i_sample = " ".join([test_token[x] for x in tmp_idx[:n_sample]])
        i_encode = bpe.bp_encode(i_sample, subword_vocab, subword_2_idx)
        
        n_input = len(i_encode)
        tmp_test_in[0, :n_input] = i_encode
        
        gen_ids = seq2seq_model.infer(
            tmp_test_in, SOS_token)
        gen_phrase = bpe.bp_decode(
            gen_ids.numpy()[0], idx_2_subword)

        print("Iteration", str(n_iter) + ":")
        print("Elapsed Time:", elapsed_tm, "mins.")

        print("Average p(z) Loss:  ", avg_pz)
        print("Average q(z|x) Loss:", avg_qzx)
        print("Average X-Ent Loss: ", avg_xent)
        print("Average Total Loss: ", avg_loss)
        print("Gradient Clip:", gradient_clip)
        print("Learning Rate:", seq2seq_optim.lr.numpy())
        
        print("")
        print("Input Phrase:")
        print(i_sample.replace("<", "").replace(">", ""))
        print("Generated Phrase:")
        print(" ".join(gen_phrase).replace("<", "").replace(">", ""))
        print("Actual Phrase:")
        print(" ".join(tmp_o_tok).replace("<", "").replace(">", ""))
        print("")
        
        # Save the training progress. #
        train_loss_list.append((n_iter, avg_loss))
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        train_loss_df.to_csv(train_loss_file, index=False)
        
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        print("-" * 50)
    
    if n_iter % cooling_step == 0:
        print("Cooling the CPU for 2 minutes.")
        time.sleep(120)
        print("Resuming training.")
