import gensim
from gensim import corpora
from gensim import similarities
from pprint import pprint

# list of app reviews
reviews = [
    "The user interface is intuitive and easy to use.",
    "I love the guided workouts feature!",
    "The app syncs seamlessly with Apple Health.",
    "The nutrition advice given is very helpful.",
    "I faced some issues with the swim tracking feature."
]

# lemmetation
texts = [[word for word in review.lower().split()] for review in reviews]

# create dict
dictionary = corpora.Dictionary(texts)

# bag-of-words
corpus = [dictionary.doc2bow(text) for text in texts]

# LDA modeling
lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

# topics
pprint(lda_model.print_topics())

# list of features
features = ["user interface", "guided workouts", "apple health", "nutrition", "swim tracking"]

# use gensim similarities module
index = similarities.MatrixSimilarity(lda_model[corpus])

# compare the similarity
for feature in features:
    feature_bow = dictionary.doc2bow(feature.lower().split())
    feature_lda = lda_model[feature_bow]
    sims = index[feature_lda]
    print(f"\nSimilarity scores for '{feature}':")
    for review, score in zip(reviews, sims):
        print(f"{review} -> {score:.2f}")