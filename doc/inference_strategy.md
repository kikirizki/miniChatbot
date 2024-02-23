# LLM inference strategy with KV-cache
Imagine that we want to do text completion using an LLM with KV-cache (there is slight difference between inferencing using KV-cache and without KV-cache). For simplicity let's pretend that each word is a token. Let say we want to predict the next 3 words that will follow this sententence.

$\fbox{what}\fbox{is}\fbox{your}\dots$

there are many strategy we can use to achieve this

## Greedy
This is the simplest strategy but in practice it often lead to bad result (we will talk about better strategy later). Since we want to predict the next 3 words then pad our prompt with 3 words 


$\fbox{what}\fbox{is}\fbox{your}\fbox{pad}\fbox{pad}\fbox{pad}$

then we input the first token to LLM

$\text{LLM}(\fbox{what})=\text{logit}_1$

$\text{logit}_1$ is the LLM prediction for the next token after $\fbox{what}$ token, but we will not use it until we reach the $\fbox{pad}$ token, we do this for filling the KV-cache. Since we use the KV-cache we will not use $\fbox{what}\fbox{is}$ sequence for the next step we will only use the last token, Because when we input the $\fbox{is}$ token, it is equal to inputing $\fbox{what}\fbox{is}$ tokens without KV-cache, the token $\fbox{what}$ is already stored in the KV-cache. So it will look like this :

$\text{LLM}(\fbox{is})=\text{logit}_2$

$\text{LLM}(\fbox{your})=\text{logit}_3$

after we reach $\fbox{pad}$ token, simply choose index of maximum value of the predicted logit as the next token, in our case $\text{logit}_3$ for 4-th token, and we use this token for generating next logit and keep doing this untill we reach the desired length.



