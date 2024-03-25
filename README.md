# NSL-test
**Self-attention** Bilibili Li Hongyi’s explanation video: https://www.bilibili.com/video/BV1Xp4y1b7ih

KV Cache introduction: https://zhuanlan.zhihu.com/p/630832593

---

This is an incomplete implementation of GPT-2. 

A quick breakdown of each of the files:

* `encoder.py` contains the code for OpenAI's BPE Tokenizer.
* `utils.py` contains the code to download and load the GPT-2 model weights, tokenizer, and hyper-parameters.
* `NSL-gpt2.py` contains the actual GPT model and generation code which we can run as a python script, but it is an incomplete version. Believe that you can successfully complete it😎👍.

#### Dependencies
```bash
pip install -r requirements.txt
```
#### Usage

The first run requires downloading the model, which is slow, please be patient.

```bash
python NSL-gpt2.py "Alan Turing theorized that computers would one day become"
```

Which generates

```
 the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

We only used the **124M** model in this test. You can also control the number of tokens to generate, the model size (one of `["124M", "355M", "774M", "1558M"]`), and the directory to save the models:

```bash
python gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --model_size "124M" \
    --models_dir "models"
```
