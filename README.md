<div style="text-align:center;">
<img src="https://github.com/kikirizki/miniChatbot/blob/main/assets/minichatbot_icon.png" width="256px" height="256px">
</div>

## miniChatbot: A Humble Exploration of Intricate Chatbot's Theoritical Foundation 

Welcome to miniChatbot! This humble repository is a labor of love aimed at providing an accessible journey through the intricate world of chatbot theoritical foundation and it's code implementation. Our goal is simple: to share knowledge and learn together, one step at a time.

## Our Humble Objective

miniChatbot strives to be a gentle guide for those curious about chatbots, whether you're just starting your journey or have been exploring for a while. We aim to break down complex concepts into bite-sized pieces, emphasizing understanding over complexity. Our focus is on:

    Taking small steps, one iteration at a time
    Exploring the math and theory behind chatbots with humility and curiosity
    Keeping our code concise and approachable, like a friendly conversation


## How to Run The Demo
![](https://github.com/kikirizki/miniChatbot/blob/main/assets/demo.gif)
currently the code contain the simple re-implementation LLaMA2 emphasizing code readability over maintainability and robustness, hence the code is different from offical llama2 model, but still compatible with it's official checkpoint, please refer to the official repo to get the model checkpoint

To run the chatbot demo please run the following command

```bash
python3 demo.py [model name (mistral/llama)] [path to checkpoint directory]  [path to tokenizer weight] [optional, allow gpu or not setting to 0 will force to use CPU]
```
examples
```bash
python3 -m demo llama ~/LLaMa2/7B/  ~/LLaMa2/tokenizer.model
```
```bash
python3 -m demo mistral ~/mistral-7B-v0.1/ ~/mistral-7B-v0.1/tokenizer.model
```
to download mistral weight you can download from the offical mistral website here 
https://docs.mistral.ai/models/

to download llama2 weight you can, please head to this llama2 official repo https://github.com/facebookresearch/llama#download and <b>click request a new download link</b>


## The Philosophy

In the spirit of humility, we've miniChatbot is build upon these simple philosopical pillars:

- Iteration: Follow along as we build our chatbot step by step, learning and growing with each iteration.

- Mathematics: Delve into the mathematical underpinnings of our chatbot with humility and awe.

- Papers: Explore research papers and academic articles with us, marveling at the wisdom of those who came before.

- Code: Dive into our humble codebase, where simplicity and clarity reign supreme.


## Table of Contents
- Introduction to Large Language Model
  - Tokenization
    - BPE
  - Transformer neural network architecture
    - scaled dot product attention
    - positional embedding

  - LLM inference strategy
    - greedy
    - random sampling
      - top-k
      - top-p   
- LLaMa2
  - RMS Normalization
  - Rotary Embedding
  - KV-Cache
  - Grouped Query Attention
  - [inference strategy with KV-cache](doc/inference_strategy.md)
- Mistral
  - Sliding Window Attention
  - Sparse Mixture of Experts
  - KV-Cache with rolling buffer
  - inference strategy with rolling buffer KV-cache
- Finetuning
  - LoRA: Low-Rank Adaptation of Large Language Models 
  - qLoRA  

## Todo list

- [x] re-implement llama2 inference with simplicity and ease of understanding in mind
- [x] write simplest working code of chatbot demo
- [x] re-implement mistral inference with simplicity and ease of understanding in mind
- [ ] explain the idea behind sliding self attention 
- [ ] write the math derivation of rotary embedding and it's code implementation
- [ ] explain the ide behind RMS normalization paper
- [ ] explain the idea behind SwiGLU activation function
- [ ] write a brief explanation about transformer architecture
- [ ] explain probability theory behind the scaled dot product denumerator 
- [ ] implement RAG
- [ ] explain RAG and it's theoritical understanding
- [ ] write a gentle introduction to prompt engineering
- [ ] implement LoRA
- [ ] explain LoRA paper


## We Humbly Wellcome You to Collaborate

miniChatbot is a humble endeavor, and we welcome collaboration from all who approach with humility and respect. Whether you're a seasoned expert or a humble novice, your contributions are valued and appreciated. For guidance on how to humbly contribute, please consult our CONTRIBUTING.md file.

## Humbly Seeking Feedback


Your feedback is essential in our quest for continuous improvement. If you have suggestions, gentle critiques, or words of encouragement, please share them with us by opening an issue or reaching out directly.

## License

miniChatbot is licensed under the MIT License, a humble gesture allowing for the free exchange of knowledge and ideas.



References:
1. This code is inspired by [Andrej Kharpaty great lecture] (https://www.youtube.com/watch?v=PaCmpygFfXo&t=1274s)
2. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser,
and Illia Polosukhin. Attention is all you need, 2017
3. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://z-p3-scontent.fcgk9-1.fna.fbcdn.net/v/t39.2365-6/10000000_662098952474184_2584067087619170692_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=40Urht-bf6sAX_7tNsr&_nc_ht=z-p3-scontent.fcgk9-1.fna&oh=00_AfBM6t7ZeFdtOr6Dixq8qjDQMcDGMNb8DssqyMkZyvyzSQ&oe=65D7367F)
4. [LLaMA2 official code](https://github.com/facebookresearch/llama)
5. [Mistral arxiv pre-print](https://arxiv.org/abs/2310.06825)
