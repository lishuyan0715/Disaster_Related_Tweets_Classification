# Disaster_Related_Tweets_Classification

## Run my workflow by using `Makefile`

In the contemporary world, everyone has a smartphone, which offers incredible convenience for people’s communication. Based on the ubiquitousness of smartphones, Twitter, one of the most significant social media that users can communicate and share their lives, enable users to announce disasters in real-time. 
However, it’s not quite easy to identify whether a person reports a real disaster or just use disaster-like words metaphorically. For example, people can use “earthquake” to describe the real earthquake, but they can also use it to depict an amazing discovery. Therefore, what I want to do is to build machine learning and Natural Language Processing (NLP) models that predict which Tweets are about real disasters and which ones are not.  

### Download Trained Model 
Since training the BERT model will spend more than 10 hours, I highly recommend users to download the pre-trained BERT model

Download from the link: https://drive.google.com/file/d/15HOZY12i3Cvusl1XwucXiQpFzcisncPq/view?usp=sharing

Store the downloaded file to the models folder. 

### Run the EDA
```bash
 make eda
```

### Run the data processing
```bash
 make data_process
```

### Run the experiments
```bash
 make experiments
```

### Run the BERT model
```bash
 make train
```

### Run the API
```bash
 make api
```