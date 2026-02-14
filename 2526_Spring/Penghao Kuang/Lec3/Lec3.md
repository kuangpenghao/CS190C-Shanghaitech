---

marp: true
theme: default
paginate: true
math: katex
style: |
    section {
        background: linear-gradient(
            to bottom,
            #a9deff 0%,
            #ffffff 15%,
            #fdfdfd 95%,
            #acd3ff 100%
        ) !important;
        color: #0c0000;
    }

---

# CS190C Lec3
Iterations of Transformer

---

## Overview

* Architecture
* Attention
  * KV Cache
  * RoPE
  * Sliding window/Sparse attention
* FFN
  * SwiGLU
  * MoE
* Normalization
  * Normalization Position
  * Normalization Methods

---

## PART1: Architecture

---

## 1.Encoder model

<div style="display: flex;">

<div style="flex: 0.4; padding-right: 5px;">

<p align="center">
    <img src="2.png" width=300>
</p>

</div>

<div style="flex: 1; padding-left: 5px;">

One of encoder models is BERT. It is actually the encoder part of Original Transformer. https://arxiv.org/abs/2104.09864

* **What can it do**: Given a masked sentence `Andrew Ng [MASK] an ML researcher.` It can infer the embedding of all `[MASK]`s.
* **How to train it**: `MLM` (masked language model) method: given sentences with masks at a certain ratio. Compare its output with standard answer for `[MASK]`
* Good at tasks about understanding the text, but poor in generating.

</div>

</div>

---

## 2.Decoder model

<div style="display: flex;">

<div style="flex: 1.2; padding-right: 5px;">

<p align="center">
    <img src="4.png" width=650>
</p>

</div>

<div style="flex: 1; padding-left: 5px;">

<br>

One of decoder models is GPT-3.  https://arxiv.org/abs/2005.14165

It is the right part of the image: 
* Remove encoder of Original Transformer.
* Remove cross-attention of Original Transformer.

</div>

</div>

---

## 2.Decoder model

<div style="display: flex;">

<div style="flex: 0.3; padding-right: 5px;">

<p align="center">
    <img src="3.png" width=200>
</p>

</div>

<div style="flex: 1; padding-left: 5px;">

* **What can it do**: Given a sentence, and it can autoregressively decode: 
  * Predict next token based on the given sentence now.
  * Add predicted token to "given sentence" then.
* **How to train it**: `CLM` (causal language model) method: given an input sentence and an output sentence. Compare the output of model based on input sentence with the standard output sentence. (More in Lec 6)
* Good at generating. But perform not as good as Encoder for understanding tasks. (Only unidirectional attention)

</div>

</div>

---

## 3.Encoder-Decoder model

Similar to Original Transformer. One of this kind of model is T5. https://arxiv.org/abs/1910.10683

Good at tasks like Seq2Seq. But this kind of models are now almost completely replaced by Decoder models.

* Training method: mask a part of sentence and try to fix it. It is not similar to "prompt+answer" format (infer task), need further finetuning.
* For certain parameter amount, a considerable part lies in encoder. But we only need one time of encoding, and `seq_len` times of decoding. Wasted.
* Should cache multiple groups of KVs, especially cross-attention is more complex to maintain. Cost significantly more memory.

---

## For architecture……

We will focus more on Decoder model in this course, and it is truly the most common type of architecture nowadays.

<p align="center">
    <img src="8.png" width=575>
</p>

---

## PART2: Attention

---

## PART2.1: RoPE

https://arxiv.org/abs/2104.09864

---

## Position embedding

Original Transformer use sine position encoding, but as we discussed in Lec2, it has several disadvantages, especially the problem of absolute position index and problems of very long sentences in inference cause by it.

Most LLMs nowadays use RoPE as position embedding method.

---

## RoPE

$\vec{x} \Rightarrow R_i\vec{x}$ ($\vec{x} \in R^d$,$d$ is even)

Divide the d-dimensional vector into several sub-segments of length 2, resulting in a total of $d/2$ sub-segments, each sub-segment makes a rotation of angle $\theta_{i,k}$ . (Proportional to position $i$)

$$
R^i = 
\begin{bmatrix}
R^i_1 & 0 & 0 & \cdots & 0 \\
0 & R^i_2 & 0 & \cdots & 0 \\
0 & 0 & R^i_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & R^i_{d/2}
\end{bmatrix}
\quad
R^i_k = 
\begin{bmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{bmatrix}.
$$

where  $\theta_{i,k} = \frac{i}{\Theta^{2k/d}}$ 

---

<style scoped>
div > div {
  font-size: 0.73em;
}
</style>

## RoPE

$$
R^i = 
\begin{bmatrix}
R^i_1 & 0 & 0 & \cdots & 0 \\
0 & R^i_2 & 0 & \cdots & 0 \\
0 & 0 & R^i_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & R^i_{d/2}
\end{bmatrix}
\quad
R^i_k = 
\begin{bmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{bmatrix}.
$$

<div style="display: flex;">



<div style="flex: 1; padding-left: 5px;">

The transformation caused by position, is only related to relative position.

Property of matrix  $R$ :  $R^{m}(R^n)^T = R^{m-n}$ (Transpose is inverse, multiply is add in power)

In the attention mechanism, for the query vector  $q_i$  at position  $i$  and the key vector  $k_j$  at position  $j$ :
* $q'_i = R^iq_i,\ k'_j = R^jk_j$
* $(q'_i)^T k'_j = q_i^T (R^{i})^T (R^j) k_j^T = q_i^T R^{j-i} k_j$

</div>

<div style="flex: 0.35; padding-right: 0px;">

<div style="display: flex; align-items: center; height: 100%;">
    <p align="center" style="width: 100%;">
        <img src="11.png">
    </p>
</div>

</div>

</div>

---

## How to understand it?

$(q'_i)^T k'_j = q_i^T (R^{i})^T (R^j) k_j^T = q_i^T R^{j-i} k_j$

* Rotate $p$ and $q$ with their absolute position, and pure relative rotation without any other transformation (like addition) causes relative position.
  * Review: Sine method has absolute position related terms.
* Only change direction, do not change length.
  * $\|q^Tk\|=\|q\|\cdot\|k\|\cos(\Delta\theta) \Rightarrow \text{Attention logit}$
  * If we use sine position encoding method, the influence of position embedding to length of vector is existent and uncontrollable, causing distortion in attention logits.
  * But in RoPE, same relative position and two vectors must cause the same attention logits!

---

## How to understand it?

<p align="center">
    <img src="14.png" width=1000>
</p>

---

## Part2.2: KV-Cache

---

## Autoregressive decode task

Suppose we continuation a sentence: "There are three……"

What should we calculate in attention?

* $q$ vector of token in position to predict.
* All $k$,$v$ vectors of former tokens:`There` `are` `three`.

Suppose we decode "kinds". What should we calculate in attention next turn?

* $q$ vector of token in new position to predict.
* All $k$,$v$ vectors of former tokens:`There` `are` `three` `kinds`.

We observe that we calculate $k$,$v$ vectors of `There` `are` `three` repeatedly. Can we cache it?

---

## KV-Cache

We call this method "KV-Cache": Cache all former $k$,$v$ vectors, so that time complexity is reduced from $\Theta(N^2)$ to $\Theta(N)$ for vectors calculation.

However, if we try to cache all vectors, it may cost an extremely large amount of memory. Can we try to optimize it, or make a compromise between accuracy and memory?

There are several methods: `MQA` `GQA` `MLA` and so on.

---

## MQA

<div style="display: flex;">

<div style="flex: 0.5; padding-right: 5px;">

<p align="center">
    <img src="5.png" width=250>
    <img src="6.png" width=250>
</p>

</div>

<div style="flex: 1; padding-left: 5px;">

MQA (Multi-Query Attention)

* Baseline: Normal MHA.
* MQA: For multiple heads of queries, share one group of values and keys vectors.

Pros and cons?

* Greatly saves memory space.
* However, queries of different heads can only refer to one channel of information, leading to accuracy reduction.

</div>

</div>

---

## GQA

<div style="display: flex;">

<div style="flex: 0.5; padding-right: 5px;">

<p align="center">
    <img src="5.png" width=250>
    <img src="7.png" width=250>
</p>

</div>

<div style="flex: 1; padding-left: 5px;">

<br>

GQA (Grouped-Query Attention)

* Baseline: Normal MHA.
* GQA: Separate all heads into $G$ groups, each group share the same values and key vectors.

Make a balance between original MHA and MQA. Numerous LLMs are using this method.

</div>

</div>

---

## A new idea

* Maybe we can avoid directly store vectors, because $k$,$v$ of all heads usually not full rank combining together.
* So we can try to compress $k$,$v$ of all heads into a small dense vector, and try to recover $k$,$v$ when needed. That is: we just need to store the small vectors, and need a little more calculation of restoration.
* This is Multi-Head Latent Attention (MHA) proposed in Deepseek-V2.

https://arxiv.org/abs/2104.09864

---

<style scoped>
div > div {
  font-size: 0.9em;
}
</style>

## MLA

<div style="display: flex;">

<div style="flex: 0.5; padding-right: 5px;">

<p align="center">
    <img src="9.png" width=400>
</p>

</div>

<div style="flex: 1; padding-left: 5px;">

* Compress: $c_{KV,t}=x_tW_{DKV}$. From $R^{d_{model}}$ to $R^{d_c}$
* We can restore $k$ in head $h$: $k_t^{(h)}=W^{h}_{UK}$, same for $v$: $v_t^{(h)}=c_{KV,t}W^{(h)}_{UV}$
* So $\text{Score}^{(h)}=q_i^{(h)}(k_j^{(h)})^T=q_i^{(h)}(c_{KV,t}W^{(h)}_{UK})^T=[q_i^{(h)}(W^{(h)}_{UK})^T]c_{KV,j}^T$, $q_i^{(h)}(W^{(h)}_{UK})^T$ is fixed for certain token, do not need to calculate for each position.
* So for score calculation, we just need to store $c_{KV,j}$.
* For $\text{output}^{(h)}=\sum_j \text{Weights}^{(h)}\cdot v_j^{(h)}$ $=\sum_j \text{Weights}^{(h)}\cdot c_{KV,j}W^{(h)}_{UV}$
* We just need to calculate $\sum_j \text{Weights}^{(h)}\cdot c_{KV,j}$, and multiply $W^{(h)}_{UV}$ at last. Still need $c_{KV,j}$ only.

</div>

</div>

---

<style scoped>
div > div {
  font-size: 0.9em;
}
</style>

## MLA


<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">

Anything wrong?

$\Rightarrow$ We haven't consider position embeddings!

Can it still solid given RoPE applied?

* $(qR_n)(kR_m)^T=qR_nR_m^TW_{UK}^Tc_{KV}^T$

We find $R_nR_m$ is determined dynamically by position.

That is: For certain token, we cannot just calculate one vector! It is not solid for RoPE!

</div>

<div style="flex: 1; padding-left: 5px;">

We use **Decoupled RoPE**. That is: calculate score of $q,k$ vectors and positions separately.

* $k_{C,i}^{(h)}=c_{KV,t}W^{(h)}_{UK}$
* $k_{R,i}^{(h)}=RoPE(h_iW_{KR}^{(h)})$
* $k_i^{(h)}=[k_{C,i}^{(h)},k_{R,i}^{(h)}]$
* $q_{C,i}^{(h)}=q_{i}^{(h)}$
* $q_{R,i}^{(h)}=RoPE(q_i^{(h)})$
* $q_i^{(h)}=[q_{C,i}^{(h)},q_{R,i}^{(h)}]$

</div>

</div>

---

## MLA

* $k_i^{(h)}=[k_{C,i}^{(h)},k_{R,i}^{(h)}]$
* $q_i^{(h)}=[q_{C,i}^{(h)},q_{R,i}^{(h)}]$
* $\text{Score}^{(h)}=q_i^{(h)}(k_j^{(h)})^T=q_{C,i}^{(h)}(k_{C,j}^{(h)})^T+q_{R,i}^{(h)}(k_{R,j}^{(h)})^T$<br>$=q_i^{(h)}(W_{UK}^{(h)})^Tc_{KV,j}^T+q_{R,i}^{(h)}(k_{R,j}^{(h)})^T$<br>Term1 is the same as before, term 2 needs normal calculation of RoPE.
* $\text{output}^{(h)}$ is similar as before.

---

## KV-Cache methods 

<p align="center">
    <img src="10.png" width=1200>
</p>


---

## PART2.3: Sparse attention

---

The time complexity of calculating attention is $\Theta(N^2)$. The most important cost is calculation of $QK^T$.

So, can we optimize the calculation method to achive a banlance between accuracy and time efficiency?

We try to calculate sparse attention. That is: only calculate a part of logits according the designed rules.

---

## Sliding window attention

* We only "care" about the nearest $k$ tokens. That is: we only calculate attention score in a band that is not so wide.
* Since Transformer has multilayers, information of older tokens absorbed in bottom layers will carried in tokens in the sliding window, so all tokens can absorbed all former tokens' information indirectly.

<p align="center">
    <img src="12.png" width=600>
</p>

---

<style scoped>
div > div {
  font-size: 0.9em;
}
</style>

## Sparse Transformer

https://arxiv.org/abs/1904.10509

<div style="display: flex;">

<div style="flex: 0.8; padding-right: 5px;">

<div style="display: flex; align-items: center; height: 100%;">
    <p align="center" style="width: 100%;">
        <img src="13.png" width=600>
    </p>
</div>

</div>

<div style="flex: 1; padding-left: 5px;">

* Sparse Transformer (strided)
  * Caring about recent toknes and periodic former tokens, just like "intensive reading" and "roughly review", suitable for data approximately periodic.
* Sparse Transformer (fixed)
  * Caring about modular recent tokens(such as: this line) and modular former tokens (such as: certain column), suitable for data with approximately chunk structures. 

</div>

</div>

---

## PART3: FFN

---

## PART3.1: SwiGLU

---

## Original Transformer FFN

$\text{FFN}(x)=\sigma(xW_1)W_2$, $W_1\in R^{d_{model}*4d_{model}}$, $W_2\in R^{4d_{model}*d_{model}}$

What's the activation function $\sigma$?

In the original transformer and other early LLMs (like T5), $\text{ReLU}$ is often used.

$
\sigma(x) =
\left\{
\begin{array}{l}
x, x\geq 0 \\
0, x<0
\end{array}
\right.
$

---

## Pros and cons?

* Easy.
* But for gradient calculation:<br>$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial \text{ReLU}(x)}\cdot \frac{\partial \text{ReLU}(x)}{\partial x}$<br>If $x<0$, $\frac{\partial \text{ReLU}(x)}{\partial x}=0$, the gradient is 0! This means parameters cannot update! Also, if $x=0$, it is non-differenciable.
* So it has the problem of efficient updating.

---

## SiLU / Swish

<div style="display: flex;">

<div style="flex: 0.6; padding-right: 5px;">

<p align="center">
    <img src="15.png" width=400>
</p>

</div>

<div style="flex: 1; padding-left: 5px;">

$\text{SiLU}(x)=x\cdot \sigma(x)$. 
$\sigma$ is sigmoid function: $\sigma(x)=\frac{1}{1+e^{-x}}$

* Smooth and differenciable at $x=0$.
* Avoid the problem of gradient $=0$.

$\text{Swish}(x)=x\cdot \sigma(\beta x)$
So SiLU is a special example of Swish.

</div>

</div>

All commom activation functions: https://vitalab.github.io/blog/2024/08/20/new_activation_functions.html

---

## Is it enough?

**Forward analysis**: We try to discover what's the activation elements of FFN like.

$y_i=\sigma(\sum_j x_j(W_1)_{ji})$

We find it a combination of first order interaction terms. It is hard to explicitly capture higher order interactions.

---

## Is it enough?

**Backward analysis**: We try to calculate how gradients of this network change.

$y=\sigma(xW_1)$
$\Rightarrow \frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}=(\frac{\partial L}{\partial y} \odot \sigma'(z))\cdot W^T$  (Tips:$z=xW_1$)

For a considerable part of activation functions, $0<\sigma'(z)<1$. So if $\sigma'(z)<<1$ or number of layers is large, the gradient is easy to vanish.

---

## GLU

There's a way to solve this problem: Build another path also derive from the input $x$, that is a gate. Use a function $\sigma$ to activate the values of gate, and do element-wise multiplication with $xV$.

That is:

* $y=(xV) \odot \sigma(xW_1)$ (That is: linear path and activated gate)
* $\text{output}=yW_2$

Compared with original FFN, one gate is added to dynamically control the activations of input.

We call it gate activation. This structure is called GLU.

---

## GLU

Different activation function $\sigma$ determine different kinds of GLUs.

* $\text{FFN}_{\text{ReGLU}}(x, W_1, V, W_2) = (\text{ReLU}(xW_1) \odot xV) W_2$
* $\text{FFN}_{\text{GeGLU}}(x, W_1, V, W_2) = (\text{GeLU}(xW_1) \odot xV) W_2$
* $\text{FFN}_{\text{SwiGLU}}(x, W_1, V, W_2) = (\text{Swish}(xW_1) \odot xV) W_2$

Many modern LLMs, such as LLAMA, are using SwiGLU as FFN.

So what's the advantage of them?

---

## Forward analysis

$y=(xV) \odot \sigma(xW_1)$
$\Rightarrow y_i=\text{Swish}(\sum_kx_kW_{k,i})(\sum_jx_jV_{j,i})$

That is a combination of second order interaction terms ($c_{ij}x_ix_j$). Comparing with original FFN, it can explicitly modeling higher order interactions.

---

## Backward analysis

$y=(xV) \odot \sigma(xW_1)$
$\Rightarrow \frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}=\frac{\partial L}{\partial y}[(\sigma(z)V^T)+(xV)\odot \sigma'(z)W_1^T]$

For $\sigma(z)V^T+(xV)\odot \sigma'(z)W_1^T$:
* Term2 may also vanished because $\sigma'(z)$ may be small.
* But term 1 is depended on value of activated gate $\sigma(z)$: if the gate is open, the gradients will flows through.

So it avoid gradient vanishing, and parameters' updating is controlled by the gate. 

---

## New scale

Comparing with the original FFN, there exist a new learnable matrix (the gate).

In order to make number of parameters equals to original FFN (with $d_{ff}=4d_{model}$), we should change $d_{ff}$ of SwiGLU.

$d_{ff}=\frac{8}{3}d_{model}$

---

## PART3.2: MoE

---

## For an extremly large LLM

We want to design a model structure with huge number of parameters. What can we do?

* Of course you can scale up the original transformer. But one of the problems is: The larger model means the longer inference time. The model may be extremly slow because of the large scale matrix calculation.

There exist a new method: Mixture of Experts (MoE).

---

## Divide the work and be experts

In the original Transformer, all subject of tasks share one FFN. This means the single FFN should carry all kinds of modeling works.

But in real life, some heavy works always done by division of labor: Everyone does different things and excels in their own ways.

This is mixture of experts: 

* Use multiple FFN, each FFN represents a subject.
* For each token, decide which FFN to enable.
* Only use a few, even one of them.
* The model has a huge number of parameters, but the inference speed almost unchange.

---

## MoE

<div style="display: flex;">

<div style="flex: 0.8; padding-right: 5px;">

<p align="center">
    <img src="16.png" width=800>
</p>

</div>

<div style="flex: 1; padding-left: 5px;">

* For each token, enter router layer first.
* The router layer is usually a linear activation+softmax.
* Sample top k experts (k is very small) according to the distribution.
* For k FFN outputs, weight and accumulate them as the final output.
* Others remain the same.

</div>

</div>

---

## More parameters and same FLOP

<p align="center">
    <img src="17.png" width=900>
</p>



---

## PART4: Normalization

---

## PART4.1: Normalization methods

---

## Batch Norm

We use `batch_size` samplings to run a training step. This means we need to do `batch_size` forward process in parallel. For each tensor in the embedding process, we do normalization for each element based on **`batch_size` elements at the same position**.

<p align="center">
    <img src="21.png" width=350>
</p>

---

## Batch Norm

$$y_i=\frac{x_i+\mu_i}{\sqrt{\sigma_i^2+\epsilon}}\gamma_i+\beta_i$$

$\mu_i,\sigma_i$ are calculated from `batch_size` number of elements at the same position.

We use learnable $\gamma$ and $\beta$ to restore the power of representation (For example, some features should be really big to activate the network).

---

## Batch Norm

It is used more frequently in CV, but not in LLM.

* If batch size is small, for example, equals to 1 during inference, it is very unstable.
* Cause distortion of $\mu,\sigma$ for sentences with varying length (if we use padding mask).
  * We will initially let the embedding of padding mask be $0$ or a certain vector, and it has no actually mean, which should not be included in calculation.

---

## Layer Norm

Compared with Batch Norm: we use `d_model` number of elements at the same tensor to calculate $\mu_i,\sigma_i$. Others are the same.

$$y_i=\frac{x_i+\mu_i}{\sqrt{\sigma_i^2+\epsilon}}\gamma_i+\beta_i$$

Used in the original Transformer and some other LLMs, such as BERT, GPT-2.

<p align="center">
    <img src="21.png" width=350>
</p>

---

## Root Mean Square Layer Norm (RMSNorm)

https://arxiv.org/abs/1910.07467

What we do in Layer Norm?
* Re-centering (minus $\mu$)
* Re-scaling (divide  $\sigma$)

Through experiments, it turns out that Re-centering is not so important. Since normalization module will be called multiple times in forward process, can we adjust it to save FLOPs?

---

## Root Mean Square Layer Norm (RMSNorm)

$$y_i=\frac{x_i}{\text{RMS}(x)}g_i$$
$$\text{RMS(x)}=\sqrt{\frac{1}{d}\sum_ix_i^2+\epsilon}$$

* We need to walk through the tensor twice to calculate $\mu$ and $\sigma$ in LN, but once in RMSNorm.
* Performance almost do not declined, even better in some conditions.

---

## PART4.2: Normalization position

---

## Post-norm

<p align="center">
    <img src="18.png" width=600>
</p>

$$x_{t+1}=\text{Norm}(x_t+F_t(x_t))$$

That is: Normalization after calculating residual connection.

Used in the original transformer and BERT.

---

## Post-norm

But actually, this normalization is not so stable during training process.

$\frac{\partial L}{\partial x_t}=\frac{\partial L}{\partial x_{t+1}}\frac{\partial x_{t+1}}{\partial x_t}=\frac{\partial L}{\partial x_{t+1}}J_{norm,l}(I+\frac{\partial F_t}{\partial x_t})$

This means we will multiply $J_{norm,l}$ continuously, which means the hidden danger of vanishing or exploded gradient.

Almost all LLMs now are using pre-norm.

---

## Pre-norm

$$x_{t+1}=x_t+F_t(\text{Norm}(x_t))$$

<p align="center">
    <img src="19.png" width=600>
</p>


$\frac{\partial L}{\partial x_t}=\frac{\partial L}{\partial x_{t+1}}\frac{\partial x_{t+1}}{\partial x_t}=\frac{\partial L}{\partial x_{t+1}}(I+\frac{\partial F_t}{\partial x_t})$.

The form of this expression is similar to residual connection, and reduce the danger of gradient problems.

---

## Summary of Transformer architecture and modules

<div style="display: flex;">

<div style="flex: 1.8; padding-right: 5px;">

<p align="center">
    <img src="20.png" width=600>
</p>

</div>

<div style="flex: 1; padding-left: 5px;">

<br><br>

* Most use RMSNorm
* Most use Pre-Norm
* Most use RoPE
* Most use SwiGLU

</div>

</div>