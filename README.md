# NLP-Transformer-VAE
This repository contains an implementation of [Variational AutoEncoders](https://arxiv.org/abs/1312.6114) on the [Transformer](https://arxiv.org/abs/1706.03762) network. The encoder will take in a shuffled and randomly sampled input and compute the latent state before proceeding to decode the sequence.

## Training the Model
The program takes in the [Fraser short jokes](https://huggingface.co/datasets/Fraser/short-jokes) dataset. To generate the sub-word tokens, run the script
```
python process_fraser_jokes_subword.py
```
followed by
```
python train_fraser_jokes_sw_tf_ver2_transformer_vae.py
```
to train the model.

## Inference
Run the script
```
python infer_fraser_jokes_sw_tf_ver2_transformer_vae.py
```
to perform inference. Some examples include (with 7000 updates):
```
Enter prompt: how did the chicken
Input Phrase:
how did the chicken
Generated Phrase:
SOS how did the chicken cross the road ? he was chicken . EOS 
--------------------------------------------------
Enter prompt: a man walks into a bar
Input Phrase:
a man walks into a bar
Generated Phrase:
SOS what do you get when you walks into a bar ? a chair EOS 
--------------------------------------------------
```

## Interpolation in the latent space
Like most Variational AutoEncoders (VAEs), running the script
```
python interpolate_fraser_jokes_sw_tf_ver2_transformer_vae_v1.py
```
generates the output samples by interpolating in the latent space. An example is (with 7000 updates):
```
Enter 1st input: how did the chicken
Enter 2nd input: how did the frog
--------------------------------------------------
Generated Phrase:
SOS how did the chicken cross the road ? to get to the other side . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the chicken cross the road ? with a pickle . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the chicken cross the road ? with a chicken . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the chicken cross the road ? he was a little chicken . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the chicken get his girlfriend ? he was in a pair of cents . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the frog get his girlfriend ? he was in the front of the chicken . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the frog get his car ? he got pissed . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the frog get his girlfriend ? he got pissed . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the frog get his girlfriend ? he was in front of the frog . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the frog get into the car ? he got pissed . EOS 
--------------------------------------------------
Generated Phrase:
SOS how did the frog get his car ? he was frog in the front of the frog . EOS 
--------------------------------------------------
```
In the above example, the decoder generates 10 linearly interpolated samples between the latent spaces corresponding to `how did the chicken` and `how did the frog`.


