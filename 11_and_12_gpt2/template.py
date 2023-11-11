import numpy as np
from tqdm import tqdm

from encoder import get_encoder
from tools import get_params

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    softmax_x = exp_x / exp_x.sum(axis=1, keepdims=True)
    return softmax_x

def attention(Q, K, V):
    M = (Q @ np.transpose(K)) / np.sqrt(K.shape[1])
    a = softmax(M) @ V
    return a

def masked_attention(Q, K, V, mask):
    M = ((Q @ np.transpose(K)) / np.sqrt(K.shape[1]) + mask)
    a = softmax(M) @ V
    return a

def linear_projection(x, w, b):
    projection = x @ w + b
    return projection

def multi_head_attention(x, attn, number_of_heads):
    w_1, b_1 = attn["c_attn"]["w"], attn["c_attn"]["b"]
    w_2, b_2 = attn["c_proj"]["w"], attn["c_proj"]["b"]
    mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    lp = linear_projection(x, w_1, b_1)
    m = lp.shape[1]
    if np.mod(m, 3) == 0:
        d = m / 3
        split = np.hsplit(lp, 3) 
        Q = split[0]
        K = split[1]
        V = split[2]
    Q_split = np.hsplit(Q, number_of_heads)
    K_split = np.hsplit(K, number_of_heads)
    V_split = np.hsplit(V, number_of_heads)  
    for i in range(number_of_heads):
        x_new = masked_attention(Q_split[i], K_split[i], V_split[i], mask)
        if i == 0:
            x = x_new
        else:
            x = np.concatenate([x,x_new], axis=1)
    x = linear_projection(x, w_2, b_2)
    return x

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def layer_normalization(x, g, b, eps=1e-5):
    mu = np.mean(x, axis=1, keepdims=True)
    sig = np.var(x, axis=1, keepdims=True)
    eps = 1e-5
    N = (x - mu) / (np.sqrt(sig + eps))
    h = g * N + b
    return h

def feed_forward_network(x, mlp):
    w_1, b_1 = mlp["c_fc"]["w"], mlp["c_fc"]["b"]
    w_2, b_2 = mlp["c_proj"]["w"], mlp["c_proj"]["b"]
    lp = linear_projection(x, w_1, b_1)
    g = gelu(lp) 
    x = linear_projection(g, w_2, b_2)
    return x

def transformer_block(x, block, number_of_heads):
    mlp, attn = block["mlp"], block["attn"]
    ln_1, ln_2 = block["ln_1"], block["ln_2"]
    g_1, b_1, g_2, b_2 = ln_1["g"], ln_1["b"], ln_2["g"], ln_2["b"]
    ln = layer_normalization(x,g_1, b_1)
    mha = multi_head_attention(ln, attn, number_of_heads)
    x_mha = x + mha
    ln = layer_normalization(x_mha, g_2, b_2)
    x_feed = feed_forward_network(ln, mlp)
    x = x_mha + x_feed
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, number_of_heads):
    g_final, b_final = ln_f["g"], ln_f["b"]
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, block, number_of_heads)
    x = layer_normalization(x, g_final, b_final)
    return x @ wte.T

def generate(input_text, tokens_to_generate=40, model_size="124M", models_dir="models", loading_bar=True):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    hparams, params = get_params(model_size, models_dir)
    encoder = get_encoder(model_size, models_dir)
    number_of_heads = hparams["n_head"]
    max_context = hparams["n_ctx"]
    input_ids = encoder.encode(input_text)
    assert len(input_ids) + tokens_to_generate < max_context
    output_ids = []
    if loading_bar:
        loop_range = tqdm(range(tokens_to_generate), "Thinking...")
    else:
        loop_range = range(tokens_to_generate)
    for _ in loop_range:
        output = gpt2(input_ids + output_ids, **params, number_of_heads=number_of_heads) 
        next_id = np.argmax(output[-1])
        output_ids.append(int(next_id))

    output_text = encoder.decode(output_ids)
    return output_text

if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    
    Test your implemetntation with something like this:
    print(generate("Hello! How do you do?"))

    You can try out different sized models from this list: ["124M", "355M", "774M", "1558M"]
    Make sure you have enough space on your device since the bigger models are quite large.
    """
    print(generate("What is the best food in the world?"))
    print(generate("What is the most watched television show?"))
    print(generate("What is the meaning of life?"))
