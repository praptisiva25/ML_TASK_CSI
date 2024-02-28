from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
nltk.download('wordnet')


def preprocess_text(text):

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)


def cluster_texts(texts, num_clusters):
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words='english', max_df=0.5, min_df=0.1)
    X = vectorizer.fit_transform(texts)
    
    
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())
    
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
    kmeans.fit(X)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red')
    plt.title('Clusters')
    plt.show()
    
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(texts[i])
    return dict(clusters)


def identify_food_poisoning_cluster(clusters):
    food_poisoning_cluster = None
    for label, texts in clusters.items():
        for text in texts:
            if 'food poisoning' in text.lower():
                food_poisoning_cluster = texts
                break
        if food_poisoning_cluster:
            break
    return food_poisoning_cluster


def combine_texts(food_poisoning_texts, additional_inputs):
    combined_texts = food_poisoning_texts + additional_inputs
    return combined_texts


def generate_project_brief(combined_texts):
   
    project_brief = '\n'.join(combined_texts)
    return project_brief


texts = [
    "I got food poisoning from the restaurant yesterday. It was awful.",
    "The symptoms of food poisoning include nausea, vomiting, and diarrhea.",
    "Last week, several customers reported cases of food poisoning after dining at our restaurant.",
    "Food poisoning is a serious health issue that requires immediate attention.",
    "The CDC has issued a warning about a potential outbreak of food poisoning in our area.",
    "I suspect I have food poisoning after eating at that sketchy food truck.",
    "The hospital admitted several patients with severe food poisoning symptoms.",
    "Food poisoning incidents have been on the rise in recent months."
]


preprocessed_texts = [preprocess_text(text) for text in texts]


clusters = cluster_texts(preprocessed_texts, num_clusters=3)


food_poisoning_texts = identify_food_poisoning_cluster(clusters)


additional_inputs = [
    "We need to address the issue of food safety in our restaurant.",
    "Foodborne illnesses can have serious consequences for our customers and reputation.",
    "Implementing proper hygiene and sanitation measures is crucial for preventing food poisoning."
]


combined_texts = combine_texts(food_poisoning_texts, additional_inputs)

# Generate project brief
project_brief = generate_project_brief(combined_texts)

print("Generated Project Brief:")
print(project_brief)
