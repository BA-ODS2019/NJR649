import json
import requests
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.probability import FreqDist

# This creates two subdirectories called "theguardian" and "collection"
ARTICLES_DIR = join('theguardian', 'collection')
makedirs(ARTICLES_DIR, exist_ok=True)


# Change this for your API key:
MY_API_KEY = '6dad8792-4499-43fd-a8db-65cd421da27a'

API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {
    'from-date': "", # leave empty, change start_date / end_date variables instead
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'api-key': MY_API_KEY
}

# day iteration from here:
# http://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates

# Update these dates to suit your own needs.
start_date = date(2019, 6, 1)
end_date = date(2019,11, 1)

dayrange = range((end_date - start_date).days + 1)
for daycount in dayrange:
    dt = start_date + timedelta(days=daycount)
    datestr = dt.strftime('%Y-%m-%d')
    fname = join(ARTICLES_DIR, datestr + '.json')
    if not exists(fname):
        # then let's download it
        print("Downloading", datestr)
        all_results = []
        my_params['from-date'] = datestr
        my_params['to-date'] = datestr
        current_page = 1
        total_pages = 1
        while current_page <= total_pages:
            print("...page", current_page)
            my_params['page'] = current_page
            resp = requests.get(API_ENDPOINT, my_params)
            data = resp.json()
            all_results.extend(data['response']['results'])
            # if there is more than one page
            current_page += 1
            total_pages = data['response']['pages']

        with open(fname, 'w') as f:
            print("Writing to", fname)

            # re-serialize it for pretty indentation
            f.write(json.dumps(all_results, indent=2))

# Update to the directory that contains your json files
# Note the trailing /
directory_name = "theguardian/collection/"

ids = list()
texts = list()
sections = list()
for filename in os.listdir(directory_name):
    if filename.endswith(".json"):
        with open(directory_name + filename) as json_file:
            data = json.load(json_file)
            for article in data:
                id = article['id']
                fields = article['fields']
                text = fields['bodyText'] if fields['bodyText'] else ""
                ids.append(id)
                texts.append(text)
                section = article['sectionId']  # ID name of each article
                sections.append(section)  # Adding each ID name to a list.

categories = set(sections)
# Changes the 'sections' list into a set, meaning that title dublicates is eliminated.

print(categories)
# Here we print the 'categories' i.e. sectionIds, to identify which sections should be included in our collection.
# Looking at the sectionIds, we'll limit our collection articles with the following sectionIds:
# Society, media, business, us-news, australia-news, world, law, global, global-development, politics, news, uk-news.
# As our analysis revolves around which topics was discussed in regards to 'authoritarianism' in politics, we choose
# the headers most relevant to said analysis. We do this to avoid categories such as books, film, artanddesign, games,
# fashion, music, etc., all of which will be sorted out.

# Before we do this processing though, let's do some analysis of our current data set, to create a point of comparison.
# Here we calculate the number of characters in our current dataset.
total_characters = list()
for text in texts:
    total_characters.append(len(text))

# Here we calculate the number of words (+ unique words) in our current dataset.
total_words = list()
for text in texts:
    words = text.split()
    total_words.extend(words)
unique_words = set(total_words)

# Now let's create the subset of our dataset, using the previously mentioned categories that we've selected.
idxs = list()
subtexts = list()
for i, section in enumerate(sections):
    if section in ['society', 'media', 'business', 'us-news', 'australia-news', 'world', 'law', 'global',
                   'global-development', 'politics', 'news', 'uk-news']:
        idxs.append(i)
        subtexts.append(texts[i])
print('Number of files under selected categories: %s' % len(idxs))

# Now we can do a bit of comparison, to see what our pre-processing achieved.
# Let's apply the same calculations as earlier, to our processed dataset

new_total_characters = list()
for subtext in subtexts:
    new_total_characters.append(len(subtext))

new_total_words = list()
for subtext in subtexts:
    new_words = subtext.split()
    new_total_words.extend(new_words)
new_unique_words = set(new_total_words)

# Let's then take a look at our statistics vs our new

print('Total number of characters in dataset before and after pre-processing: ' +
      str(sum(total_characters)) + ' (before) ' + str(sum(new_total_characters)) + ' (after).')

# Let us also import some stopwords for filtering, to further refine our dataset.

model_vect = CountVectorizer(stop_words='english',
                             token_pattern=r'[a-zA-Z\-][a-zA-Z\-]{2,}')

# Here we import the CountVectorizer feature from sklearn, which will let us create a model vectorizer with our own
# parameters e.g. filtering out stopwords.

data_vect = model_vect.fit_transform(subtexts)
# Then I apply the model to my data, consisting of .json files from The Guardian's API.

print("Shape: (%i, %i)" % data_vect.shape)
# Here we print out the shape of our newly created document-term matrix, to get an idea of the shape. he first number
# indicates the number of rows i.e. the number of texts, and the second number describes the number of combined terms
# said documents.

# Now, after getting rid of the stop words, let's see what our top 10 most used words, look like.
counts = data_vect.sum(axis=0).A1
top_idxs = (-counts).argsort()[:10]
# First we grab the 'counts' i.e. the number of times a given term appears in all our documents, and then we list
# the top-10 sorted by their counts and listed as their indexes.

inverted_vocabulary = dict([(idx, word) for word, idx in model_vect.vocabulary_.items()])
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
# Now we use an inverted vocabulary to get the word for each of the indexes in our previously constructed top-10 list.

print("Top words: %s" % top_words)
# Lastly, we print our list. Here we see that we have successfully gotten rid of the stop words, and that the words
# in the list seems very reasonable, considering their origin.

# Now it's time for select a number of articles for our analysis, using a query. Let's define our terms:
terms = ['authoritarian', 'authoritarianism', 'autocrat', 'autocratic']
term_idxs = [model_vect.vocabulary_.get(term) for term in terms]
term_counts = [counts[idx] for idx in term_idxs]
print(term_counts)

# Now let's calculate the term weights
model_tfidf = TfidfTransformer()
data_tfidf = model_tfidf.fit_transform(data_vect)
idfs = model_tfidf.idf_
term_idfs = [idfs[idx] for idx in term_idxs]
print(term_idfs)

# Now let's perform a serach using our query items
query = " ".join(terms)
print(query)

# Now let's transform our query string into a query vector
query_vect_counts = model_vect.transform([query])
query_vect = model_tfidf.transform(query_vect_counts)

# Here we sort the documents according to which ones fit our query terms the best
sims = cosine_similarity(query_vect, data_tfidf)
sims_sorted_idx = (-sims).argsort()

# Now we create a dataframe which ranks the 13290 documents of our subset according to their content's similiarty to our
# query terms. After we've done that, we print out the top 100 documents to hone in on the documents that really matter.
print("Shape of 2-D array sims: (%i, %i)" % (len(sims), len(sims[0, :])))
query_df = pd.DataFrame(data=zip(sims_sorted_idx[0, :], sims[0, sims_sorted_idx[0, :]]),
                        columns=["index", "cosine_similarity"])
print("Shape of query dataframe " + str(query_df.shape))

# Now we filter our dataframe so we are only left with documents with a cosine similarity above 0, i.e. documents in
# which any of our query terms are actually mentioned. We do this by checking each columns with a boolean condition.
is_above_zero = query_df['cosine_similarity'] > 0.000000000
print(is_above_zero)

# Then we create a new dataframe, only including those values that are above 0, as can be seen by the amount of rows
# remaining in our new dataframe.
query_df_above_zero = query_df[is_above_zero]
print(query_df_above_zero.shape)

# Finally, let's visually the top words in our final dataset, to answer our reserach question
model_lda = LatentDirichletAllocation(n_components=4, random_state=0)
data_lda = model_lda.fit_transform(data_vect)

# Here we get the top words for 4 of the topics
for i, term_weights in enumerate(model_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = ["%s (%.3f)" % (model_vect.get_feature_names()[idx], term_weights[idx]) for idx in top_idxs]
    print("Topic %d: %s" % (i, ", ".join(top_words)))

# Lastly we make 4 wordclouds, to better illustrate the size of the top words
for i, term_weights in enumerate(model_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = [model_vect.get_feature_names()[idx] for idx in top_idxs]
    word_freqs = dict(zip(top_words, term_weights[top_idxs]))
    wc = WordCloud(background_color="white",width=300,height=300, max_words=10).generate_from_frequencies(word_freqs)
    plt.subplot(2, 2, i+1)
    plt.imshow(wc)
