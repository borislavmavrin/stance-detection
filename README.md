# stance-detection
Code for my MSc thesis: Statistical Modeling Of Stance Detection.

## Abstract
In recent years fake news has become a more serious problem. This is mainly due to the popularity of social networks, search engines and news aggregators that propagate fake news. Classifying news as fake is a hard problem. However, it is possible to distinguish between fake and real news, by considering how many related tweets agree/disagree with the news. Therefore, in the simplest case the problem can be reduced to identifying whether a given tweet agrees with, disagrees with or is unrelated to the news in question. In general, this problem is referred to as ’stance detection’. In machine learning terminology this is a classification problem. This thesis investigates more advanced Natural Language Models, such as matching Long Short Term Memory model and soft attention mechanism applied to stance detection problem. The ideas are tested using a publicly available data set.

## Long Short Term Memory (LSTM)
LSTM is a non-linear hidden time-series model. It is based on Recurrent Neural Network (RNN). The basic idea is similar to the Hidden Markov Model, the observable time-series is modelled by hidden state representaion.
RNN can be defined inductively in the following way

Given a time-series {x_1, x_2, ... x_T}:
 1. Intialize hidden state: h_0 = 0
 2. Update h_1 = f(W_h * h_0 + W_x * x_1)
 3. Fit the model by predicting x_2: min (W * h_1 - x_2) with respect to W_h, W_x, W.

W_h, W_x, W are estimated weight matrices, f is an activation function, usually sigmoid or tangh. Note that W_h, W_x and W are the same for each step in the time series.

However, estimation of RNN is complicated by the Vanishing/Exploding gradient problem. During gradient descent updates the gradient might vanish or diverge to infinity. The gradient has the form of g^T, i.e. some expression raised to the power T (length of the time series sequence). Hence, if g becomes small, the gradient g^T vanishes, and if g is big enough, g^T explodes.
LSTM introduced gating mechanism which mitigates the problem. In short LSTM might 'forget' the h_t or x_t. LSTM dynamically changes importance weight of h_t and x_t at each step. A very good introduction can be found here: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Soft attention
Soft attention arose from the alignment problem in machine translation. 
For example, given a sentence in English 

{e_1, e_2, e_3} 

and its corresponding translation in French 

{f_1, f_2, f_3, f_4}

the problem is to align groups of corresponding words, i.e. (e_1 ~ f_1, f_2, f_3), (e2, e3 ~ f_4). This correspondence can be represented by 3x4 matrix with 3 rows for English words and 4 columns for French words:

[[1, 1, 1, 0]

[0, 0, 0, 1]

[0, 0, 0, 1]]

Soft attention mechanism tries to learn these correspondence by modeling the weighting matrix in a soft way. Soft here means that the weights are not necessarily 0, 1, but rather from 0 to 1. This allows the model to be end-to-end differentiable  and thus can be estimated by gradient descent. This attention mechanism can be easily adapted to a more general class of time-series problems.

### For more details see the thesis (file inside the repo): Mavrin_Borislav_201709_MSc.pdf

## How to run code:
### 1. Install and activate virtual environment:
 ```bash
 cd stance-detection
 pip2 install virtualenv
 virtualenv -p python2 .env
 source .env/bin/activate
 ```
### 2. Install dependencies:
 ```bash
 pip2 install -r requirements.txt
 ```
### 3. Install stopword corpus into home folder:
```bash
python2 -c "import nltk nltk.download('stopwords')"
```
### Note: installing modules from requirements.txt is crucial since the TensoFlow API was changing a lot at the time of the creation of the code.
