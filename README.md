# Tweet2Story



*Tweet2Story* is a framework to **automatically extract narratives** from sets of Tweets. Despite the focus on tweets, it can also be used to extract narratives from regular texts. The tool performs a set of Natural Language Processing tasks, in order to transform the texts it receives into [brat style annotations](https://brat.nlplab.org/standoff.html).

The tool was tested on a certain set of tweets retrieved from the [Signal1M-tweetir](https://research.signal-ai.com/datasets/signal1m-tweetir.html) dataset. This dataset consists on sets of tweets about an event **linked** to a news article about the same event. The tests consisted on empirical evaluation and on comparing the narrative from the tweets - extracted with Tweet2Story - with the narrative from the news article - annotated by an expert linguist.

Taking this into account, this repository provides **two** things:

* [The Tweet2Story framework.](#how-to-use-tweet2story)
* [The news articles annotated by the expert linguist (gold annotations).](#gold-annotation-dataset)



## How to use Tweet2Story

Placeholder

## Gold annotations dataset

### Definition

-----

The gold annotations dataset contains a set of **48 news articles** and **564 tweets** retrieved from the [Signal1M-tweetir](https://research.signal-ai.com/datasets/signal1m-tweetir.html) dataset. The articles are exclusively in **English** and as stated by [Signal](https://research.signal-ai.com/datasets/signal1m.html), they are collected from a variety of news sources. Each article was curated by a pre-processing pipeline that removed unnecessary parts of the text, such as links. Finally, the texts were annotated by an expert linguist using the [brat style annotations](https://brat.nlplab.org/standoff.html).

### Dataset Structure

-------------------

The gold annotations dataset, we named it NE-Twitter (NE for Narrative Extraction) for simplicity, has a simple structure. It contains a folder with the brat style annotated narratives from each article and a csv file with the tweets IDs linked to their respective news article ID, both about the same event.

![](dataset-structure.PNG) 



### Why should you use it

-----

There are no other dataset that we know of (as of August 25, 2021) that provide annotated news articles by an expert linguist in the brat format.

The automatic extraction of narratives is a current task where you can use this dataset. The annotations could be used as input and training data for models, or their structure can be used to make new custom annotations adapted to your needs. On another hand, the comparison of the narrative of news articles about an event with social media posts about the same event is also an interesting study and one can use these annotations to compare the narratives from both sides.



### How to reproduce the dataset

---

The dataset is created through the [Signal1M](https://research.signal-ai.com/datasets/signal1m.html) dataset where we get the news articles and by the [Twitter API](https://developer.twitter.com/en/docs/twitter-api) where we get the tweets. To retrieve the data from the Signal1M dataset, it is best to follow the steps in this [tutorial](https://github.com/signal-ai/Signal-1M-Tools), which uses Elasticsearch to collect the news articles (since the original dataset is quite large in size).

The tweets can be collected by hydrating them from the Twitter API. Simply install the [Twarc](https://github.com/DocNow/twarc) utility using pip and run the following command, where the "tweetIDs.txt" contains a list of the list of tweet IDs in the dataset. This command will extract the JSON response from the Twitter API.

```python
!twarc hydrate tweetIDs.txt > tweets.jsonl
```

Note: You will need to have a Twitter API token and a Twitter API secret to be able to use the Twitter API and twarc.



### Contact

For further information related to this dataset please contact Vasco Campos using the e-mail address <vasco.m.campos@inesctec.pt>.

