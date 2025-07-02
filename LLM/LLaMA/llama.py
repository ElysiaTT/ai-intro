import d2l
import torch
import numpy
import pandas
@torch.inference_mode()  # 使用torch.inference_mode()装饰器，使得该函数在推理模式下运行，以提高推理性能
def generate(
    self,
    prompt_tokens: List[List[int]],  # 输入的提示词的token列表，每个提示词被表示为一个整数列表
    max_gen_len: int,  # 生成文本序列的最大长度
    temperature: float = 0.6,  # 用于控制采样随机性的温度值，默认为0.6
    top_p: float = 0.9,  # 用于核采样的top-p概率阈值，默认为0.9
    logprobs: bool = False,  # 指示是否计算token的对数概率，默认为False
    echo: bool = False,  # 指示是否在生成的输出中包含提示词的token，默认为False
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    """
    Generate text sequences based on provided prompts using the language generation model.

    Args:
        prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
        max_gen_len (int): Maximum length of the generated text sequence.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
        Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

    Note:
        This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    params = self.model.params  # 获取模型的参数
    bsz = len(prompt_tokens)  # 计算提示词的批次大小
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)  # 断言批次大小不超过模型的最大批次大小

    min_prompt_len = min(len(t) for t in prompt_tokens)  # 计算提示词中最短的长度
    max_prompt_len = max(len(t) for t in prompt_tokens)  # 计算提示词中最长的长度
    assert max_prompt_len <= params.max_seq_len  # 断言最长提示词长度不超过模型的最大序列长度
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)  # 计算生成序列的总长度

    pad_id = self.tokenizer.pad_id  # 获取分词器的填充token的id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")  # 创建一个填充好的tensor，用于存储生成的token
    for k, t in enumerate(prompt_tokens):  # 遍历每个提示词
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")  # 将提示词的token填充到对应的位置
    if logprobs:  # 如果需要计算对数概率
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)  # 创建一个与tokens形状相同的tensor，用于存储对数概率

    prev_pos = 0  # 初始化前一个位置为0
    eos_reached = torch.tensor([False] * bsz, device="cuda")  # 创建一个布尔tensor，用于标记每个序列是否达到结束符
    input_text_mask = tokens != pad_id  # 创建一个掩码tensor，标记哪些位置不是填充token
    if min_prompt_len == total_len:  # 如果最短提示词长度等于总长度
        logits = self.model.forward(tokens, prev_pos)  # 前向传播计算logits
        token_logprobs = -F.cross_entropy(
            input=logits.transpose(1, 2),
            target=tokens,
            reduction="none",
            ignore_index=pad_id,
        )  # 计算token的对数概率

    for cur_pos in range(min_prompt_len, total_len):  # 从最短提示词长度开始遍历到总长度
        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)  # 前向传播计算当前位置的logits
        if temperature > 0:  # 如果温度值大于0
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)  # 对logits进行softmax计算概率
            next_token = sample_top_p(probs, top_p)  # 使用top-p采样获取下一个token
        else:  # 如果温度值为0
            next_token = torch.argmax(logits[:, -1], dim=-1)  # 取概率最大的token作为下一个token

        next_token = next_token.reshape(-1)  # 将next_token的形状调整为一维
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )  # 如果当前位置是提示词的一部分，则保持不变，否则使用采样得到的token
        tokens[:, cur_pos] = next_token  # 更新tokens中的当前位置
        if logprobs:  # 如果需要计算对数概率
            token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens[:, prev_pos + 1 : cur_pos + 1],
                reduction="none",
                ignore_index=pad_id,
            )  # 计算当前位置的对数概率
        eos_reached |= (~input_text_mask[:, cur_pos]) & (
            next_token == self.tokenizer.eos_id
        )  # 更新eos_reached，标记哪些序列达到了结束符
        prev_pos = cur_pos  # 更新前一个位置
        if all(eos_reached):  # 如果所有序列都达到了结束符
            break  # 跳出循环

    if logprobs:  # 如果需要计算对数概率
        token_logprobs = token_logprobs.tolist()  # 将token_logprobs转换为列表
    out_tokens, out_logprobs = [], []  # 初始化输出的token列表和对数概率列表
    for i, toks in enumerate(tokens.tolist()):  # 遍历每个生成的token序列
        # cut to max gen len
        start = 0 if echo else len(prompt_tokens[i])  # 如果echo为True，则从0开始，否则从提示词的长度开始
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]  # 截取到最大生成长度
        probs = None  # 初始化对数概率为None
        if logprobs:  # 如果需要计算对数概率
            probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]  # 截取对应的对数概率
        # cut to eos tok if any
        if self.tokenizer.eos_id in toks:  # 如果生成的token序列中包含结束符
            eos_idx = toks.index(self.tokenizer.eos_id)  # 找到结束符的位置
            toks = toks[:eos_idx]  # 截取到结束符之前
            probs = probs[:eos_idx] if logprobs else None  # 如果需要计算对数概率，也截取到结束符之前
        out_tokens.append(toks)  # 将截取后的token序列添加到输出列表中
        out_logprobs.append(probs)  # 将截取后的对数概率添加到输出列表中
    return (out_tokens, out_logprobs if logprobs else None)  # 返回生成的token序列和对数概率（如果需要）


# 定义一个名为text_completion的方法，用于文本补全
def text_completion(
    self,
    prompts: List[str],  # 输入的文本提示列表，每个元素是一个字符串
    temperature: float = 0.6,  # 控制采样随机性的温度值，默认为0.6
    top_p: float = 0.9,  # 核采样的top-p概率阈值，默认为0.9
    max_gen_len: Optional[int] = None,  # 生成的补全序列的最大长度，可选参数
    logprobs: bool = False,  # 是否计算token的对数概率，默认为False
    echo: bool = False,  # 是否在生成的输出中包含提示词的token，默认为False
) -> List[CompletionPrediction]:
    """
    Perform text completion for a list of prompts using the language generation model.

    Args:
        prompts (List[str]): List of text prompts for completion.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
        List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

    Note:
        This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    # 如果max_gen_len没有提供，则将其设置为模型的最大序列长度减1
    if max_gen_len is None:
        max_gen_len = self.model.params.max_seq_len - 1
    # 将每个提示文本编码为token列表，存储在prompt_tokens中
    prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    # 调用generate方法生成token和对数概率
    generation_tokens, generation_logprobs = self.generate(
        prompt_tokens=prompt_tokens,  # 输入的提示词token列表
        max_gen_len=max_gen_len,  # 生成的最大长度
        temperature=temperature,  # 温度值
        top_p=top_p,  # top-p阈值
        logprobs=logprobs,  # 是否计算对数概率
        echo=echo,  # 是否包含提示词token
    )
    # 如果需要计算对数概率
    if logprobs:
        # 返回包含生成文本、token列表和对数概率的字典列表
        return [
            {
                "generation": self.tokenizer.decode(t),  # 解码生成的token为文本
                "tokens": [self.tokenizer.decode(x) for x in t],  # 解码每个token为文本
                "logprobs": logprobs_i,  # 对应的对数概率
            }
            for t, logprobs_i in zip(generation_tokens, generation_logprobs)  # 遍历生成的token和对数概率
        ]
    # 如果不需要计算对数概率，只返回包含生成文本的字典列表
    return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

# 定义一个名为chat_completion的方法，用于聊天补全
def chat_completion(
    self,
    dialogs: List[Dialog],  # 输入的对话列表，每个对话是一个消息列表
    temperature: float = 0.6,  # 控制采样随机性的温度值，默认为0.6
    top_p: float = 0.9,  # 核采样的top-p概率阈值，默认为0.9
    max_gen_len: Optional[int] = None,  # 生成的响应序列的最大长度，可选参数
    logprobs: bool = False,  # 是否计算token的对数概率，默认为False
) -> List[ChatPrediction]:
    """
    Generate assistant responses for a list of conversational dialogs using the language generation model.

    Args:
        dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

    Returns:
        List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

    Raises:
        AssertionError: If the last message in a dialog is not from the user.
        AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

    Note:
        This method generates assistant responses for the provided conversational dialogs.
        It employs nucleus sampling to introduce controlled randomness in text generation.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    # 如果max_gen_len没有提供，则将其设置为模型的最大序列长度减1
    if max_gen_len is None:
        max_gen_len = self.model.params.max_seq_len - 1
    # 存储提示词的token列表
    prompt_tokens = []
    # 存储不安全请求的标志列表
    unsafe_requests = []
    # 遍历每个对话
    for dialog in dialogs:
        # 检查对话中是否包含特殊标签，如果包含则标记为不安全请求
        unsafe_requests.append(
            any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
        )
        # 如果对话的第一条消息是系统消息
        if dialog[0]["role"] == "system":
            # 重新组合对话，将系统消息和第一条用户消息合并
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        # 断言对话中的角色顺序必须是系统（可选）、用户、助手交替出现
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        # 编码对话中的消息为token列表，并求和得到对话的token列表
        dialog_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    bos=True,
                    eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],  # 提取用户消息
                    dialog[1::2],  # 提取助手消息
                )
            ],
            [],
        )
        # 断言对话的最后一条消息必须是用户消息
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        # 编码对话的最后一条用户消息为token列表，并添加到对话的token列表中
        dialog_tokens += self.tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            bos=True,
            eos=False,
        )
        # 将对话的token列表添加到提示词的token列表中
        prompt_tokens.append(dialog_tokens)

    # 调用generate方法生成token和对数概率
    generation_tokens, generation_logprobs = self.generate(
        prompt_tokens=prompt_tokens,  # 输入的提示词token列表
        max_gen_len=max_gen_len,  # 生成的最大长度
        temperature=temperature,  # 温度值
        top_p=top_p,  # top-p阈值
        logprobs=logprobs,  # 是否计算对数概率
    )
    # 如果需要计算对数概率
    if logprobs:
        # 返回包含生成文本、token列表和对数概率的字典列表
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t)
                    if not unsafe
                    else UNSAFE_ERROR,  # 如果请求不安全，返回错误信息
                },
                "tokens": [self.tokenizer.decode(x) for x in t],  # 解码每个token为文本
                "logprobs": logprobs_i,  # 对应的对数概率
            }
            for t, logprobs_i, unsafe in zip(
                generation_tokens, generation_logprobs, unsafe_requests
            )  # 遍历生成的token、对数概率和不安全请求标志
        ]
    # 如果不需要计算对数概率，只返回包含生成文本的字典列表
    return [
        {
            "generation": {
                "role": "assistant",
                "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,  # 如果请求不安全，返回错误信息
            }
        }
        for t, unsafe in zip(generation_tokens, unsafe_requests)  # 遍历生成的token和不安全请求标志
    ]


# 定义一个名为sample_top_p的函数，用于执行top-p（核）采样
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    # 对概率分布进行排序，返回排序后的概率和对应的索引
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算排序后概率的累积和
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 创建一个掩码，用于筛选累积概率超过阈值p的部分
    mask = probs_sum - probs_sort > p
    # 将掩码对应的概率置为0
    probs_sort[mask] = 0.0
    # 对筛选后的概率进行归一化
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 从归一化后的概率分布中进行多项式采样，得到下一个token的索引
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # 根据采样得到的索引，从原始索引中获取对应的token索引
    next_token = torch.gather(probs_idx, -1, next_token)
    # 返回采样得到的token索引
    return next_token
