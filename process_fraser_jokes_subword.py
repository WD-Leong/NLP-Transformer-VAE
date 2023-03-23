
import time
import pandas as pd
import pickle as pkl
from collections import Counter
import byte_pair_encoding as bpe
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

print("Loading the data.")
start_tm = time.time()

tmp_file = "../Data/jokes/short-jokes-train.parquet"
tmp_data = pd.read_parquet(tmp_file)
max_len  = 25
print("Total of", str(len(tmp_data)), "jokes loaded.")

# Extract the data. #
tmp_jokes = []
for n_row in range(len(tmp_data)):
    tmp_joke = tmp_data.iloc[n_row]["text"]
    tmp_joke = tmp_joke.replace("\"", "").replace("\n", " ")
    tmp_jokes.append(tmp_joke)

# Process the data. #
tmp_jokes_filtered = []

w_counter = Counter()
for tmp_joke in tmp_jokes:
    tmp_tokens = [
        x for x in word_tokenizer(tmp_joke.lower()) if x != ""]
    
    if len(tmp_tokens) <= max_len:
        w_counter.update(tmp_tokens)
        tmp_jokes_filtered.append(tmp_joke)
    del tmp_tokens

print("Total of", str(len(tmp_jokes_filtered)), "jokes filtered.")
del tmp_jokes

word_counts = []
for word, count in w_counter.items():
    tmp_word = "<" + word + ">"
    tmp_word = "".join([x+" " for x in tmp_word]).strip()
    word_counts.append((tmp_word, count))
word_counts = dict(word_counts)

elapsed_tm = (time.time() - start_tm) / 60
print("Total of", str(len(word_counts)), "words.")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Fit the subword vocabulary. #
print("Fitting subword vocabulary.")
start_tm = time.time()

n_iters = 500
vocab_size = 8000
tuple_out  = bpe.learn_subword_vocab(
    word_counts, n_iters, vocab_size=vocab_size)

subword_vocab = tuple_out[0]
idx_2_subword = tuple_out[1]
subword_2_idx = tuple_out[2]

elapsed_tm = (time.time() - start_tm) / 60
print("Total Sub-word Vocabulary size:", 
      len(subword_vocab), "sub-word tokens.")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Encode the corpus to subword tokens. #
print("Encoding the corpus to subwords.")
start_tm = time.time()

jokes_sw_tokens = []
for n_joke in range(len(tmp_jokes_filtered)):
    tmp_joke = tmp_jokes_filtered[n_joke]
    tmp_joke_sw = bpe.bp_encode(
        tmp_joke, subword_vocab, subword_2_idx)
    
    jokes_sw_tokens.append(tmp_joke_sw)
    if (n_joke+1) % 25000 == 0:
        print(n_joke+1, "jokes processed.")

elapsed_tm = (time.time() - start_tm) / 60
print("Elapsed Time:", elapsed_tm, "mins.")

# Save the data. #
print("Saving the file.")

tmp_pkl_file = "../Data/jokes/short_jokes.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(jokes_sw_tokens, tmp_file_save)

    pkl.dump(subword_vocab, tmp_file_save)
    pkl.dump(idx_2_subword, tmp_file_save)
    pkl.dump(subword_2_idx, tmp_file_save)

