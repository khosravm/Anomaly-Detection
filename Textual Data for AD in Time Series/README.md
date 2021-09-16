<p align="justify"> Time series are present everywhere in the world since they are used to measure various phenomena (e.g., temperature, spread of a virus, sales, etc.). Forecasting of time series is highly essential and advantageous for better decision makings. However, using only the historical values of the time series is often insufficient. </p>

<p align="justify">While modern techniques are able to explore large sets of temporal data to build forecasting models, they typically neglect valuable information that is often available under the form of unstructured text. Although this data is in a radically different format, it often contains contextual explanations for many of the patterns that are observed in the temporal data. As a matter of fact, texts often contain useful information that can potentially help predict future values of many time series variables. For example, sentiment in social media about some major events may be relevant for predicting the change of next day’s stock market. So, we can construct effective additional features based on related text data for time series forecasting.
  
<p align="justify">Basically, there are some methods which present text mining techniques to combine textual data with time-series. Generally and from a data fusion perspective, combining time-series data with textual information for better understanding real-world phenomena is a very important, yet challenging, problem. As mentioned above, the key intuition is that the textual information could contain clues that correlate with the time-series observations and, at least to some extent, explain their behavior.
 
<p align="justify">Data fusion approaches are often divided into three main categories:
  
  (i) Approaches that treat different data sources equally and put together the features extracted from them into one single feature vector. 
  
  (ii) Approaches that use different sources of data at different stages 
  
  (iii) Approaches that feed different datasets into different parts of a model simultaneously. 
  
<p align="justify">The model, we proposed here belongs to 3rd category. In fact, the proposed approach make use of document embeddings and convolutional layers to capture patterns in the text that correlate with the time-series observations, which are modeled either using LSTMs or a stack of fully-connected layers. This architecture is able to construct latent representations of the time series and textual data, which can then be combined to produce more accurate forecasts.
