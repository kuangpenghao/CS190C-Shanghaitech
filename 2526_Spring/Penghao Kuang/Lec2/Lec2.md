---
marp: true
theme: default
paginate: true
math: true
---

# CS190C Lec2
From Seq2Seq to Transformer

---

## Overview 
* What is Seq2Seq
* Seq2Seq with RNN
* Seq2Seq with Transformer
  * Attention
  * Position Embedding
  * Feed-Forward Network
  * Residual Connection
  * Complexity analysis
* Thinking

---

## PART1: What is Seq2Seq

---

## Examples

* Given a Chinese sentence, translate to English $\Rightarrow$ Machine translation
* Given a long paragraph, conclude to a short sentence $\Rightarrow$ Abstract writing
* Given an academic report, convert to easy-to-understand article $\Rightarrow$ Style transfer

### Conclusion

Seq2seq is a kind of language task to input a sequence and output a sequence according to certain demand.

---

## PART2: Seq2Seq with RNN

---

## Review

<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">

* At each time step, receive a word and encode it.
* Mix the encoded word and former text information.
* At each time step, the information (embedding) of text will be enriched once.
* At last time step, the embedding of text will contain the whole information of all words.

</div>

<div style="flex: 1; padding-left: 5px;">

<img src="1.png">

</div>

</div>

---

## A naive idea

* The embedding of text will contain the whole information of all words at last time……
* Input a sequence and use RNN to process it……
* The last text embedding $h^T$ is hoped to carry all information of it.
* We can further decode output sequence based only on $h^T$.
* We use another RNN to decode, right after the first RNN.

---

## Seq2Seq with RNN

* Encoder and Decoder.
<p align="center">
    <img src="2.png" width="800">
</p>

---

## Pros and Cons?

* Easy and intuitive.
* RNN: Gradient Problems.

If the sequence is too long, the model's performance is not so good——not only gradient problems……

* Encode everything into a single vector without proper weights.
* Forward:$h^t=\sigma(W_hh^{t-1}+W_ee^t+b_1)$, remote information will receive significantly more changes——Although they may be important

`The writer of the books (is/are?)`$\Rightarrow$ `books` is closer and `writer` is farther.

---

## PART3: Seq2Seq with Transformer

https://arxiv.org/abs/1706.03762

## PART3.1：Attention

---

## How to encode with proper weight?

`The boy who is picking apples (is/are)?`

For this word, it should find words whose properties has strong correlationship with its "problems".

* How to describe properties of each words' "problems"?
* How to describe properties of each words themselves?
* There's no doubt that: If "problems" and "properties" has strong relationship, it will cause significantly high weight, like obvious attention.
* If a word find another word significantly satisfied the requirements above, how to describe the information the word provide?

---

## How to encode with proper weight?

* All above should rely on the words' embeddings.

If a word has embedding $x \in R^d$, each $x_i$ ($1\leq i \leq d$) may stands for the significance level of a certain semantic feature……

* The whole semantic of a word can be seen as conbination of all semantic features and their significance level
* Each semantic feature may represent certain kind of problems, properties and information

---

## How to encode with proper weight?

* For all $d$ semantic features, we hope to train a matrix $W^Q \in R^{d_{qk}*d}$, which means: Certain semantic feature will cause a "problem", whose embedding $\in R^{d_{qk}}$

* Similarly, we hope to train a matrix $W^K \in R^{d_{qk}*d}$, which means: Certain semantic feature will represent a property, whose embedding $\in R^{d_{qk}}$.

* Similarly, we also hope to train a matrix $W^V \in R^{d_{v}*d}$, which means: Certain semantic feature will feedback certain kind of information, whose embedding $\in R^{d_{v}}$.

---

## How to encode with proper weight?

So, how do we calculate the "problem" of a word, which is the combination of semantic features?

Recall: Certain value of $x_i$ means the significance level of i-th semantic features.

$W^Q=[w^q_1 w^q_2 …… w^q_d]$.
$q=x_1w^q_1+x_2w^q_2+……+x_dw^q_d$
$\Rightarrow q=W^Q x$, which is the "problem" of a word.

Similarly:$k=W^K x, v=W^V x$

$x\Rightarrow q,k,v$

---

## Scaling Dot Product Attention (SDPA)

So, if word $i$ with "problem" $q_i$, word $j$'s weight score with word $i$ is $q^T_ik_j$. Use softmax to normalize all words weight with word $i$ to get weight distribution, which is the weight of sum $v_j$ feedback to word $i$.

Write to matrix formular:
$\text{Attention}_i=\text{Softmax}(\frac{Q^TK}{\sqrt{d_{qk}}})V$.

Tips:$\sqrt{d_{qk}}$ is the scaling factor to ensure Softmax logits not too large or too small, thus ensure the distribution not degenerate to one-hot or uniform.

---

## Multihead Attention (MHA)

Does a semantic feature have only one "problem", only one "property"?

Maybe we can repeat SDPA in $H$ channels, each channel stands for certain kind of "problem" and "property" pair. So word $i$ has: $h_1$,$h_2$,……,$h_H$ ($h_j=\text{Attention}_i^j$).

Final $\text{Attention}_i=\text{Concat}(h_1,h_2,……,h_H)W^O$

---

## PART3.2： Position Embeddings

---

## A Problem

* In real-world contexts, if two sentences have same word but different orders, they almost certainly has different meanings.  `I study LLM` `LLM study I`

* But what in MHA?

Only word embeddings and their $q,k,v$ matters, do not include position.

$\Rightarrow$ We should add some "tag" for each word, representing their position in the sentence. Thus it will have an impact on MHA.

---



---

## PART3.3: Feed-Forward Network

---



---

## PART3.4: Residual Connection

---



---

## PART3.5: Complexity

---



---

<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">



</div>

<div style="flex: 1; padding-left: 5px;">



</div>

</div>