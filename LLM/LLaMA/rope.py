import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.axes import Axes

def create_sin_cos_cache(max_num_tokens, head_size):
    theta = 10000 ** (-np.arange(0, head_size, 2) / head_size)
    theta = theta.reshape(-1, 1).repeat(2, axis=1).flatten()

    pos = np.arange(0, max_num_tokens)
    table = pos.reshape(-1, 1) @ theta.reshape(1, -1)  # [max_num_tokens, head_size]

    sin_cache = np.sin(table)
    sin_cache[:, ::2] = -sin_cache[:, ::2]

    cos_cache = np.cos(table)
    return sin_cache, cos_cache

def rotate_half(vec):
    return vec.reshape(-1, 2)[:, ::-1].flatten()

def rotary(vec, pos, sin_table, cos_table):
    return vec * cos_table[pos] + rotate_half(vec) * sin_table[pos]

def plot(plt_obj: Axes, pic_index, query_index=0, head_size=256, max_num_tokens=8192, step=1):
    q_vec = np.ones(head_size)
    k_vec = np.ones(head_size)
    sin_table, cos_table = create_sin_cos_cache(max_num_tokens, head_size)

    rotated_q_vec = rotary(q_vec, query_index, sin_table, cos_table)
    k_indices = np.arange(0, max_num_tokens, step)
    rotated_k_vecs = rotary(k_vec, k_indices, sin_table, cos_table)
    attn_scores = (rotated_k_vecs @ rotated_q_vec) / np.sqrt(head_size)

    plt_obj.plot(k_indices, attn_scores)
    plt_obj.set_title(f"Figure {pic_index}: query_index={query_index}, head_size={head_size}")
    plt_obj.set_xlabel("key index")
    plt_obj.set_ylabel("attention score")

plt.rcParams.update({
    "font.sans-serif": ["Times New Roman", ],
    "font.size": 10
})

_, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
plot(axes[0, 0], 1, query_index=0, max_num_tokens=512)
plot(axes[0, 1], 2, query_index=256, max_num_tokens=512)
plot(axes[1, 0], 3, query_index=0, max_num_tokens=65535)
plot(axes[1, 1], 4, query_index=0, head_size=8, max_num_tokens=65535)
plt.show()