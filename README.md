# MachineLearningGroupProject

The continuous growth of internet users has resulted in a massive volume of data flow on the Internet. This has also led to an upsurge in online criminal activities. Cyber criminals use the anonymity offered by VPN’s and Tor Browser to carry out sophisticated attacks on organisations. Therefore, it is very important to closely monitor inbound/outbound network traffic to detect patterns indicative of suspicious behaviour. Our team proposes an Application project which implements a machine learning solution that can help organisations classify network traffic as benign or suspicious. The solution can help in early detection of malware before an attack and detection of malicious activities for investigation after the attack.

The input to our algorithm is a CSV file containing more than 60 network traffic features such as Duration, Number of packets, Number of bytes, Length of packets, etc. calculated in both forward and backward directions. We then use logistic regressor, kNN classifier and Random Forest classifier to predict whether the network traffic is from Tor, VPN or belonging to normal traffic. 


Theese features are able to identify the patterns in the network traffic and help us categorise the traffic. Thus, our input and output are able to achieve the intended aim.
