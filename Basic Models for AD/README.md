**A case study of anomaly detection in Python**

1st: create a dummy dataset

2nd: Detecting anomalies just by Visualization

3rd: Clustering based approach for anomaly detection

    k-means clustering is a method of vector quantization (vq), originally from 
    signal processing, that aims to partition n observations into k clusters 
    in which each observation belongs to the cluster with the nearest mean 
    (cluster centers or cluster centroid), serving as a prototype of the 
    cluster.
    
4th: Anomaly detection as a classification problem

    Proximity-based anomaly detection (by use of  k-NN classification method): 
        The basic idea here is that the proximity of an anomaly data point to 
        its nearest neighboring data points largely deviates from the proximity 
        of the data point to most of the other data points in the data set.

<p align="justify"> While detecting anomalies, we almost always consider ROC and Precision as it 
gives a much better idea about the model's performance. 
An ROC curve (receiver operating characteristic curve) is a graph showing the 
performance of a classification model at all classification thresholds. This 
curve plots two parameters: True Positive Rate. False Positive Rate.
Infact, The ROC curve shows the trade-off between sensitivity (or TPR) and 
specificity (1 – FPR). Classifiers that give curves closer to the top-left 
corner indicate a better performance. ... The closer the curve comes to the 
45-degree diagonal of the ROC space, the less accurate the test. </p>

Precision talks about how precise/accurate your model is out of those predicted 
positive, how many of them are actual positive. Precision is a good measure to 
determine, when the costs of False Positive is high.
