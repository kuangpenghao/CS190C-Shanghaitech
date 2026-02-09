---
marp: true
theme: default
paginate: true
math: true
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

# CS190C Lec1

Word Embedding & Language Modeling

---

## Overview

* How to embed words
* What is language modeling
* Some naive and early language models
  * N-grams
  * RNN
  * LSTM
* Thinking

---

## PART1：How to embed words?

---

Problem: Computer cannot understand natural language……
* We should try to convert them, such as words, into digital form.
* How to represent words formally?

---

## A naive idea

* Like a dictionary, each word has its position in it
* Can we represent a word using a certain number?
  * That is: each word has its "one-to-one mapping value"
* Just like this:

<p align="center">

| **Word**   | I | love | Natural | Language | Processing | " " | . |
|------------|---|------|---------|----------|------------|-----|---|
| **Value**  | 0 | 1    | 2       | 3        | 4          | 5   | 6 |

</p>

`"I love Natural Language Processing."` $\Rightarrow$ `0,5,1,5,2,5,3,5,4,6`

---

## Mapping has its defect

* Can we have a method to infer this word's part of speech or meaning, just based on the mapping value?
* For example: If you only receive a string of numbers:`0,5,1,5,2,5,3,5,4,6`, can you successfully infer the meaning of this sentence?
* Mathematically, one-to-one mapping can be understood as: **The semantics of different words constitute a one-dimensional vector space!**
  * Each word is a 1-dim vector in this 1-dim space
  * Using 1-dim vector space to represent natural language is obviously not enough!

---

## Enlarge the dimension?

* If we use 2-dim space to represent some words?
  * x-aixs represent "area", y-axis represent "population"
  * Can we use 2-dim vectors to approximatly represent:`China` `India` `Canada` `Luxembourg`? (Plot a graph of it)
* Similarly, the higher the dimension, the richer the semantics it can represent.
* For example, GPT-3 uses 2048-dim to represent words.

---

## What is word embedding?

So far we know that:

* We can use n-dim vector to represent the meaning of the word formally.
* Each dim can be understand as a kind of semantics, it just like one more dimension of space in linear algebra.

We call vectors of word `Word Embedding`.

* Encoding words with appropriate word embeddings is one of the key points in natural language processing technology
* In later lecs, we will introduce some important encoder models.

---

## PART2: What is language modeling?

---

## Look at a LLM

<div style="display: flex;">

<div style="flex: 1; padding-right: 10px;">

<img src="1.png" width=800>

</div>

<div style="flex: 1; padding-left: 10px;">

* Input a prompt (a string of words)
* Give an answer based on the prompt

How does it generate words, finally forms an answer?

* Like normal speaking of human, each word should generate after formal words, and based on it logically.
* That is: based on old words, and generate new words.
* This called Language Modeling!

</div>

</div>

---

### Tips: Difference between `Language Model` and `Language Modeling`

* Language Model: Is a tool to generate certain answer based on prompts.
* Language Modeling: Is the methods to generate "new words" based on "old words".

So, what's the "methods"?

---

## General methods

There are maybe several words suitable to be the generated new word……

* But different word may have different levels of suitability.
* We can try to model "levels of suitability" into probabilistic distribution.
* That is, find a way to calculate the probability of generating a certain word as the "new word", based on "old words".

---

## Language Modeling Methods……

These are different models, using different certain methods to calculate the probabilistic distribution.
* N-grams
* RNN, LSTM (An optimized architecture based on RNN)
* Transformer
* GPT, BERT .etc (Based on Transformer)

---

## PART3： N-grams

---

## Another naive idea

Simply consider: If we just focus on a fix window of old words to generate a new word, it can still work at most of time.

For example:

* We only focus on latest 3 old words to generate a new word
* We call it `4-grams`, which means a fixed window contain old words + a new word has 4 words.
* Prompt: "Let's calculate simple multiplication! One times one……"

---

## A naive 4-grams

* old words focused: `One times one`
* Generate new word directly use statistical laws!
* We can use https://books.google.com/ngrams/
<img src="2.png" width=800>

---

## A naive 4-grams

<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">

<img src="2.png" width=800>

* We can try all words in vocabulary
* We can also get $P(new,old)$ (marginal probability of 4-grams)
* What we want to model is $P(new|old) \propto P(new,old)$

</div>

<div style="flex: 1; padding-left: 5px;">

* Suppose we've modeled $P(w_i|old)$
* We choose $\text{argmax}_{w_i}P(w_i|old)$ or sample words according to the distribution
* Suppose we decide `is` to be the new word
* Sentence now: `One times one is`
* Next turn: use `times one is` to generate a new word, and so on.

</div>

</div>

---

## Pros and cons?

* Actually it is quite simple, and do not need much calculation (especially model training)
* If you really use N-grams to generate text, what will happen?

---

## Generate text?

<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">

<img src="3.png">

</div>

<div style="flex: 1; padding-left: 5px;">

* For grammar……
  * Subject-predicate/verb-object…… part-of-speech collocation follows statistical rules
* But for semantics……
  * Missing of global context
  * One mistake will cause accumulation of subsequent errors

</div>

</div>

---

## PART4: Recurrent Nerual Network (RNN)

---

## How to make use of global context?

Encodding……
* We encode words as vectors……
* Can we encode text as vectors too?
* If we get the encoded vector of text, we can generate new words with "understanding" of text!
<p align="center">
  <img src="4.png" width="600">
</p>

---

## How to encode words and text?

* For words……
  * At first, each word still encoded with single number.
  * We hope to train a matrix $W_e\in R^{d*|V|}$, each column is the embedding of i-th word.
  * So for i-th word, we can get its embedding through $W_ee$, $e$ is the one-hot vector of this word (only get value 1 in i-th element) $\Rightarrow W_ee$ is the i-th column of $W_e$

---

## How to encode words and text?

* For text……
  * Like reading word by word, with each word read, the understanding of the text will be more substantial.
  * When a new word read: the "understanding" of the text will mixed with: Former understanding of the text, and the information of the new word.
  * That is: the embedding of the text in this time step should mix former embedding of text and embedding of the new word together.

---

## How to encode words and text?

Let embedding of text in step-t be $h^t$

* Inherit former embedding: $W_hh^{t-1}$. We hope to train $W_h$, which stands for how to proper inherit former information
* New word's information: $W_ee^t$
* Combine:$W_hh^{t-1}+W_ee^t+b_1$. ($b_1$ is the optional bias term)
* Add a nonlinear activation (usually use sigmoid)

---

## How to encode words and text?

<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">

* $h^t=\sigma(W_hh^{t-1}+W_ee^t+b_1)$
* Parameters to train:
  * $W_e$
  * $W_h$
  * $b_1$

</div>

<div style="flex: 1; padding-left: 5px;">

<img src="5.png">

</div>

</div>

---

## How to language modeling?

Just use a linear activation to $h^t$, generate the probability distribution of the new words!

<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">

* $\hat{y^t}=\text{Softmax} (Uh^t+b_2)$
* Parameters to train:
  * $U$: the linear activation matrix
  * $b_2$: the optional bias term

</div>

<div style="flex: 1; padding-left: 5px;">

<img src="4.png">

</div>

</div>

---

## Implement a simple RNN

https://github.com/kuangpenghao/NLP_models_by_hand/blob/main/toy_RNN.py


<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">

* 3 training sentences
* Train RNN
* Input sentence except the last word
* Hope to output the correct word

</div>

<div style="flex: 1; padding-left: 5px;">

<img src="6.png">

</div>

</div>

---

## Implement a simple RNN

* main function

```Python
if __name__=="__main__":

    config=TextRNNConfig()
    model=TextRNN(config)

    trainer=TextRNNTrainer(config,model)
    predictor=TextRNNPredictor(config,model)

    trainer.train()

    test_sentences=["i like dog", "i love coffee", "i hate milk"]
    for sentence in test_sentences:
        sentence=sentence.split()
        input_sentence=sentence[:-1]
        predicted_word=predictor.predict(input_sentence)
        print(f"input:{input_sentence},output:{predicted_word}")
```

---

## Implement a simple RNN

```Python
class TextRNNConfig:
    def __init__(self):
        self.n_hidden=5
        self.sentences=["i like dog", "i love coffee", "i hate milk"]

        word_list=' '.join(self.sentences).split()
        word_list=list(set(word_list))
        self.word_dict={w:i for i,w in enumerate(word_list)}
        self.number_dict={i:w for i,w in enumerate(word_list)}

        self.n_class=len(word_list)

        self.batch_size=2
        self.learning_rate=0.001
        self.epochs=1000
        self.interval=200
```

---

## Implement a simple RNN

```Python
class TextRNN(nn.Module):
    def __init__(self,config):
        super(TextRNN,self).__init__()
        self.config=config
        self.rnn=nn.RNN(self.config.n_class,self.config.n_hidden)
        self.W=nn.Linear(self.config.n_hidden,self.config.n_class,bias=True)

    def forward(self,X):
        X=X.transpose(0,1)
        ori_hidden=torch.zeros(1,X.shape[1],self.config.n_hidden)
        outputs,last_hidden=self.rnn(X,ori_hidden)
        output=outputs[-1]
        output=self.W(output)
        return output
```

---

## Implement a simple RNN

```Python
class TextRNNDataset(Dataset):
    def __init__(self,config):
        super(TextRNNDataset,self).__init__()
        self.config=config

    def __len__(self):
        return len(self.config.sentences)

    def __getitem__(self,idx):
        sentence=self.config.sentences[idx]
        words=sentence.split()
        words=[self.config.word_dict[i] for i in words]

        input_idx=words[:-1]
        one_hot=np.eye(self.config.n_class)[input_idx]
        input_one_hot=torch.tensor(one_hot,dtype=torch.float32)

        target_idx=words[-1]
        target_idx=torch.tensor(target_idx,dtype=torch.int64)

        return input_one_hot,target_idx
```

---

## Implement a simple RNN

```Python
class TextRNNTrainer:
    def __init__(self,config,model):
        self.config=config
        self.model=model
        self.loss_function=nn.CrossEntropyLoss()
        self.optimizer=torch.optim.SGD(model.parameters(),lr=self.config.learning_rate,momentum=0.9)

        self.datagetter=TextRNNDataset(config)
        self.dataloadder=DataLoader(self.datagetter,batch_size=self.config.batch_size,shuffle=True)

    def train(self):
        for epoch in range(self.config.epochs):
            for input_batch,target_batch in self.dataloadder:
                self.optimizer.zero_grad()
                output=self.model(input_batch)
                loss=self.loss_function(output,target_batch)
                loss.backward()
                self.optimizer.step()
            if (epoch+1)%self.config.interval==0:
                print(f"epoch:{epoch+1:04d},loss:{loss.item():.6f}")
```

---

## Implement a simple RNN

```Python
class TextRNNPredictor:
    def __init__(self,config,model):
        self.config=config
        self.model=model

    def predict(self,input_sentence):
        with torch.no_grad():
            word=[self.config.word_dict[i] for i in input_sentence]
            word=np.eye(self.config.n_class)[word]
            word=torch.tensor(word,dtype=torch.float32).unsqueeze(0)

            output=self.model(word)
            outcome=output.max(1,keepdim=False)[1]
            predicted_word=self.config.number_dict[outcome.item()]

            return predicted_word
```

---

## Pros and cons?

* Actually, it can make use of the information of the global text……
* When the text is extremely long, step $t$ is very large?

---

## Update parameters ($W_h$)

<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">

* $\frac{\partial L_T}{\partial W_h}=\sum_{t=1}^T\frac{\partial L_T}{\partial h^t}\frac{\partial h^t}{\partial W_h}$
* $\frac{\partial L_T}{\partial h^t}=\frac{\partial L_T}{\partial h^T}\frac{\partial h^T}{\partial h^{T-1}}……\frac{\partial h^{t+1}}{\partial h^t}$
* $\frac{\partial h^{t+1}}{\partial h^t}=\frac{\partial \sigma(W_hh^{t}+W_ee^{t+1}+b_1)}{\partial h^t}$ $=\sigma'(W_hh^{t}+W_ee^{t+1}+b_1)W_h$
* We usually use sigmoid function as $\sigma$. Then $\sigma'(z)=\sigma(z)(1-\sigma(z))$ $\in(0, 0.25]$
* $\frac{\partial L_T}{\partial h^t}=\frac{\partial L_T}{\partial h^T}\Pi_{k=t+1}^T[\sigma'(z_k)W_h]$

</div>

<div style="flex: 1; padding-left: 5px;">

<img src="5.png">

</div>

</div>

---

## Update parameters ($W_h$)

<div style="display: flex;">

<div style="flex: 1; padding-right: 5px;">

* $\frac{\partial L_T}{\partial h^t}=\frac{\partial L_T}{\partial h^T}\Pi_{k=t+1}^T[\sigma'(z_k)W_h]$
* If $\|W_h\|<1$, each term must $<1$ <br>$\Rightarrow \frac{\partial L_T}{\partial h^t} \to 0$, which may leads to $\frac{\partial L_T}{\partial W_h} \to 0$
* Likewise, if $\|W_h\|$ is large enough, $\frac{\partial L_T}{\partial W_h} \to \infty$
* Called vanishing gradient & exploding gradient

</div>

<div style="flex: 1; padding-left: 5px;">

<img src="5.png">

</div>

</div>

---

## PART5: Long Short Term Memory (LSTM)

---

<img src="7.png" width=1000>

---

<img src="9.png" width=1000>

---

<img src="8.png" width=1000>

---

## Thinkings

* *In N-grams, the model will crashed because of missing information of global context. Can we make N very large to solve this problem?*
