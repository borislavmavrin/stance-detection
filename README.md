# stance-detection
code for my MSc thesis: Statistical Modeling Of Stance Detection

## Abstract
In recent years fake news has become a more serious problem. This is mainly due to the popularity of social networks, search engines and news ag- gregators that propagate fake news. Classifying news as fake is a hard problem. However it is possible to distinguish between fake and real news, by consider- ing how many related tweets agree/disagree with the news. Therefore, in the simplest case the problem can be reduced to identifying whether a given tweet agrees with, disagrees with or is unrelated to the news in question. In general this problem is referred to as ’stance detection’. In machine learning termi- nology this is a classification problem. This thesis investigates more advanced Natural Language Models, such as matching Long Short Term Memory model and soft attention mechanism applied to stance detection problem. The ideas are tested using a publicly available data set.

## For more details see the thesis: Mavrin_Borislav_201709_MSc.pdf

## How to run code:
 1. cd stance-detection
 2. pip2 install virtualenv
 3. virtualenv -p python2 .env
 4. source .env/bin/activate # Activate the virtual environment
 5. pip2 install -r requirements.txt # Install dependencies
 6. python2 -c "import nltk nltk.download('stopwords')" # Install stopword corpus into home folder
 Note: installing modules from requirements.txt is crucial since the tensoflow API was changing a lot at the time of the creation of the code.
