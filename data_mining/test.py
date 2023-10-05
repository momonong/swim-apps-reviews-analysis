import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import nltk
import numpy as np
import spacy
from collections import Counter
from nltk.util import ngrams
from wordcloud import WordCloud


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")


st.title('Vinted Year in Review: 2022')

# Call all data


@st.experimental_memo
def dcaller(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    return df

vdf = dcaller('prepped_vinted.csv')

# Select box filters
# months mapping
monthlist = ['All Months: 2022', 'January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November', 'December']
monthdict = {'All Months: 2022': 1, 'January': 'Jan', 'February': 'Feb',
             'March': 'Mar', 'April': 'Apr', 'May': 'May', 'June': 'Jun',
             'July': 'Jul', 'August': 'Aug', 'September': 'Sep', 'October': 'Oct',
             'November': 'Nov', 'December': 'Dec'}

# source mapping
sourcelist = ['All Reviews', 'Online Reviews',
              'Play Store Reviews', 'App Store Reviews']
sourcedict = {'All Reviews': 1, 'Online Reviews': 'online',
              'Play Store Reviews': 'android', 'App Store Reviews': 'iphone'}


# Utility Functions

@st.cache
def sharder(df, mont=1, srce=1, senti=1):
    if mont != 1:
        df = df[df['month'] == mont]

    if srce != 1:
        df = df[df['source'] == srce]

    if senti != 1:
        df = df[df['sentiment'] == senti]

    return df


@st.cache
def concater(filelist):
    for i in filelist:
        if 'vinted' in i:
            vdf = dcaller(i)
            vdf['company'] = 'Vinted'
        elif 'depop' in i:
            ddf = dcaller(i)
            ddf['company'] = 'Depop'
        elif 'poshmark' in i:
            pdf = dcaller(i)
            pdf['company'] = 'Poshmark'
        elif 'thredup' in i:
            tdf = dcaller(i)
            tdf['company'] = 'Thredup'

    df3 = pd.concat([vdf, ddf, pdf, tdf], ignore_index=True)
    return df3


@st.cache
def dfcopier(df):
    return df.copy()


@st.cache(allow_output_mutation=True)
def sentipie(df):
    buf = io.BytesIO()
    colors = sns.color_palette('bright')[0:3]
    df.groupby('sentiment').size().plot(
        kind='pie', colors=colors, autopct='%.1f')
    plt.ylabel("")
    plt.savefig(buf, format="png")
    return buf


@st.cache(allow_output_mutation=True)
def senti_month_plot(df, x):
    buf = io.BytesIO()
    pl = sns.countplot(data=df, x=x, hue='sentiment')
    # pl.set(xlabel="Month")
    pl.set(ylabel="Count")
    plt.savefig(buf, format="png")
    return buf


# @st.cache(allow_output_mutation=True)
@st.cache(allow_output_mutation=True)
def line_plot(df, x, h=1):
    buf = io.BytesIO()
    if h == 1:
        g = sns.lineplot(x=x, y='compound', data=df, sort=False)
    else:
        g = sns.lineplot(x=x, y='compound', hue=h, data=df, sort=False)
    # g.set(xticklabels=[])
    g.set(title='Sentiment of Review')
    g.set(xlabel="Month")
    g.set(ylabel="Sentiment")
    # g.tick_params(bottom=False)
    g.axhline(0, ls='--', c='grey')
    # fig = plt.plot()
    plt.savefig(buf, format="png")
    return buf


@st.cache(allow_output_mutation=True)
def wordcloud_plot(df):
    buf = io.BytesIO()

    if type(df) == pd.core.frame.DataFrame:
        all_words_lem = ' '.join(word for word in df['review_lem'])
    else:
        all_words_lem = ' '.join([str(elem) for elem in df])

    wordcloud = WordCloud(background_color="white").generate(all_words_lem)

    plt.figure(figsize=(15, 11))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.tight_layout(pad=0)
    plt.axis('off')
    plt.savefig(buf, format="png")
    return buf

# Returns most negative or positive reviews


@st.cache
def extremes(df, negpos):
    if negpos == 'negative':
        ext = df.nsmallest(3, 'compound')[
            ["review", "compound", "source", "date"]]
    else:
        ext = df.nlargest(3, 'compound')[
            ["review", "compound", "source", "date"]]

    ext.reset_index(drop=True, inplace=True)
    rev = ext['review'][0].capitalize()
    src = ext['source'][0]
    dat = ext['date'][0].capitalize()

    if src == 'android':
        src = 'Play Store'
    elif src == 'iphone':
        src == 'App Store'
    elif src == 'online':
        src = 'Online'

    res = 'Review: \n' + rev + '\n\n Source: ' + src + '  Date: ' + dat

    return res


@st.cache
def ngramdf(df, size):
    words = ' '.join([word for word in df['review_lem']]).split()
    ngramlist = list(nltk.ngrams(words, size))
    cnt_ngram = Counter()
    for word in ngramlist:
        cnt_ngram[word] += 1
    df = pd.DataFrame.from_dict(cnt_ngram, orient='index').reset_index()
    df = df.rename(columns={'index': 'words', 0: 'count'})
    df['words'] = df['words'].apply(lambda x: ' '.join([item for item in x]))
    df = df.sort_values(by='count', ascending=False)
    df = df.head(10)
    df = df.sort_values(by='count')
    return df


@st.cache(allow_output_mutation=True)
def plotNgrams(dfdoc, colour):
    #unigrams = ngramdf(dfdoc, 1)
    bigrams = ngramdf(dfdoc, 2)
    trigrams = ngramdf(dfdoc, 3)
    buf = io.BytesIO()
    # Set plot figure size
    fig = plt.figure(figsize=(7, 20))
    # figsize=(30, 15)
    plt.subplots_adjust(wspace=.5)

    ax2 = fig.add_subplot(312)
    ax2.barh(np.arange(len(bigrams['words'])),
             bigrams['count'], align='center', alpha=.5, color=colour)
    ax2.set_title('Bigrams')
    plt.yticks(np.arange(len(bigrams['words'])), bigrams['words'])
    plt.xlabel('Count')

    ax3 = fig.add_subplot(313)
    ax3.barh(np.arange(len(trigrams['words'])),
             trigrams['count'], align='edge', alpha=.5, color=colour)
    ax3.set_title('Trigrams')
    plt.yticks(np.arange(len(trigrams['words'])),
               trigrams['words'])
    plt.xlabel('Count')
    plt.savefig(buf, format="png", bbox_inches='tight')
    return buf


@st.cache
def searcher(df, keyword):
    out1 = f'Search for "{keyword.upper()}"'
    df = df.loc[df['review'].str.contains(keyword, case=False)]
    nlp = spacy.load('en_core_web_sm')

    if not df.shape[0]:
        return f'{out1}\n> Number of reviews: 0\n'

    senti = df['compound'].sum()
    posdf = df.loc[df['sentiment'] == 'positive']
    negdf = df.loc[df['sentiment'] == 'negative']
    numrev = df.shape[0]
    if numrev <= 5:
        df5 = df.copy()
    else:
        df5 = df.head(5)

    revsumpos = posdf['review'].str.cat(sep=' ')
    revsumneg = negdf['review'].str.cat(sep=' ')
    doc_pos = nlp(revsumpos)
    doc_neg = nlp(revsumneg)
    poslist = [token.text for token in doc_pos if token.dep_ in [
        "amod", "advmod", "compound"] and token.head.text == keyword]
    neglist = [token.text for token in doc_neg if token.dep_ in [
        "amod", "advmod", "compound"] and token.head.text == keyword]

    if senti > 0:
        sentiment = 'Positive'
    elif senti == 0:
        sentiment = "Neutral"
    else:
        sentiment = 'Negative'

    out2 = f' Number of reviews: {numrev}'
    out3 = f' Overall sentiment: {sentiment}'
    out4 = df5['review'].tolist()

    return out1, out2, out3, out4, poslist, neglist


st.header("Overview")
st.sidebar.markdown("# Jump To:")
st.sidebar.markdown(
    """
- ### [Overview](#overview)
- ### [Your Strengths And Weaknesses](#what-are-your-strengths-and-weaknesses)
- ### [Search](#search)
"""
)
st.sidebar.markdown("## Filter")
with st.sidebar.form('Form15'):
    chart12vdf = dfcopier(vdf)
    chart35vdf = dfcopier(vdf)
    f15col1, f15col2 = st.columns(2)
    mnegtab = "Most Negative Comment in 2022"
    mpostab = "Most Positive Comment in 2022"

    with f15col1:
        source15 = st.selectbox('Select Source', sourcelist, key=151)

    with f15col2:
        month15 = st.selectbox('Select Month', monthlist, key=152)

    submitted15 = st.form_submit_button('Go')

st.sidebar.write(
    'Please note, the first two charts show the year-long trend, therefore are not affected by the month selection option.')
st.sidebar.markdown("# About The Author")
st.sidebar.markdown("## Adijat Ojutomori")
st.sidebar.write("I'm a Data Engineer with a background data analysis. I get excited about opportunities where I'm able to build and maintain data pipelines that have real product impact.")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/aojutomori/)")
st.sidebar.write("[Github](https://github.com/toussyn)")
if submitted15:
    mnegtab = 'Most Negative Comment | ' + source15 + ' | ' + month15
    mpostab = 'Most Positive Comment | ' + source15 + ' | ' + month15
    source15 = sourcedict[source15]
    month15 = monthdict[month15]
    chart12vdf = sharder(chart12vdf, srce=source15)
    chart35vdf = sharder(chart35vdf, srce=source15, mont=month15)

# The columns
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5a, col5b, col5c = st.columns([1, 7, 1])

with col1:
    st.write("The distribution of sentiments count for the year 2022.")
    chart1 = senti_month_plot(chart12vdf, 'month')
    st.image(chart1)
    plt.clf()

with col2:
    st.write(
        "The distribution of sentiments score for the year 2022")
    chart2 = line_plot(chart12vdf, x='month')
    st.image(chart2)
    plt.clf()


with col3:
    st.write("Percentage distribution of review sentiments.")
    chart3 = sentipie(chart35vdf)
    st.image(chart3)
    plt.clf()

with col4:
    st.write("Most common words found in the reviews.")
    chart4 = wordcloud_plot(chart35vdf)
    st.image(chart4)
    plt.clf()

with col5b:
    st.subheader("Extreme Reviews")

    extab1, extab2 = st.tabs([mnegtab, mpostab])
    extab1.write(extremes(chart35vdf, 'negative'))
    extab2.write(extremes(chart35vdf, 'positive'))

st.header('Your Strengths And Weaknesses')

col6, col7 = st.columns(2)
with col6:
    st.subheader('What Customers Love Most About You')
    st.write('Below are the top 10 bigrams and trigrams found in positive reviews')
    chart611vdfpos = sharder(chart35vdf, senti='positive')
    chart6 = plotNgrams(chart611vdfpos, 'blue')
    st.image(chart6)
    plt.clf()


with col7:
    st.subheader('What Customers Dislike About You')
    st.write('Below are the top 10 bigrams and trigrams found in negative reviews')

    chart611vdfneg = sharder(chart35vdf, senti='negative')
    chart7 = plotNgrams(chart611vdfneg, 'red')
    st.image(chart7)
    plt.clf()

st.header("Search")
st.write('Sometimes some words or phrase will come up in both positive and negative reviews. Sometimes you would like to know how users feel about a particular feature or a newly integrated process. Here, you can search for a particular word and it would return a wordcloud of modifiers used to that word in both positive and negative reviews. Give it a try in the search box.')
s_term = 'return'
text_input = st.text_input("Enter a word and press Enter")
if text_input:
    if len(text_input.split()) > 1:  # has more than 1 word
        st.warning("Please enter only one word")
    else:
        s_term = text_input

search = searcher(chart35vdf, s_term)
if type(search) == tuple:
    st.write(f'{search[0]}   |   {search[1]}   |  {search[2]}')
    # st.write(search[1])
    # st.write(search[2])
    col8, col9 = st.columns(2)
    with col8:
        if not len(search[4]):
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(
                'No modifiers found in positive reviews for the word ' + s_term.upper())
        else:
            chart8 = wordcloud_plot(search[4])
            st.write(
                'Modifiers found in positive reviews for the word', s_term.upper())
            st.image(chart8)
            plt.clf()

    with col9:
        if not len(search[5]):
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(
                'No modifiers found in negative reviews for the word ' + s_term.upper())
            st.write(' ')
            st.write(' ')
            st.write(' ')
        else:
            chart9 = wordcloud_plot(search[5])
            st.write(
                'Modifiers found in negative reviews for the word', s_term.upper())
            st.image(chart9)
            plt.clf()

    with st.expander("See Sample Reviews"):
        for h in search[3]:
            st.write(h)
            st.write(" ")
else:
    st.write(search)