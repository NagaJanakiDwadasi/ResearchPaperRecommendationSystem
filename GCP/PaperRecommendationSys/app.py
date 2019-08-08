import os
from flask import request
from flask import Flask, render_template
from flask_table import create_table, Col, Table, Col, LinkCol, ButtonCol
from flask_bootstrap import Bootstrap
import pandas as pd
import logging

logging.basicConfig(filename='paperrecom.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize

import json
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

template_dir = os.path.abspath('templates')
static_dir = os.path.abspath('static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
Bootstrap(app)


# Remove Stop Words
def findKeywords(example_sent):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(example_sent)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


# Add Keywords column to dataset
def addKeywordsToDataSet(filename):
    stop_words = set(stopwords.words('english'))
    df = pd.read_csv(filename)
    col1 = 'TitleKeywords'
    col2 = 'AbstractKeywords'
    if col1 and col2 in df.columns:
        return
    with open(filename, 'r', encoding="utf8") as csvfile:
        rows = csv.reader(csvfile)
        next(rows, None)
        newCol1 = []
        newCol2 = []
        for row in rows:
            # row[0] is title column
            sentence = row[0]
            # row[4] is abstract column
            sentence1 = row[4]
            titleKeywords = findKeywords(sentence)
            AbstractKeywords = findKeywords(sentence1)
            newCol1.append(titleKeywords)
            newCol2.append(AbstractKeywords)
    df2 = pd.DataFrame({'TitleKeywords': newCol1, 'AbstractKeywords' : newCol2})
    data = pd.concat([df, df2], axis=1)
    data.to_csv(filename,index=False)
    return


# genearteWordCloudMatrices
def generateWordCloudTitle(data):
    v = TfidfVectorizer()
    x = v.fit_transform(data['TitleKeywords'])
    wordCloud = v.get_feature_names()
    return wordCloud


def generateWordCloudMatrixTitle(data):
    v = TfidfVectorizer()
    x = v.fit_transform(data['TitleKeywords'])
    matrixdf = pd.DataFrame(x.toarray())
    return matrixdf


def generateWordCloud(data):
    v = TfidfVectorizer()
    x = v.fit_transform(data['AbstractKeywords'])
    wordCloud = v.get_feature_names()
    return wordCloud


def generateWordCloudMatrix(data):
    v = TfidfVectorizer()
    x = v.fit_transform(data['AbstractKeywords'])
    matrixdf = pd.DataFrame(x.toarray())
    return matrixdf


#finds TFIDF of a word
def findWordTFIDF(wordCloud, matrixdf, word):
    tFIDF = []
    for ind in matrixdf.index:
        for i in (i for i,x in enumerate(wordCloud) if x.lower() == word.lower()):
            rowTFIDF = matrixdf[i][ind]
            tFIDF.append(rowTFIDF)
    #finaldf = pd.DataFrame(tFIDF)
    return tFIDF


#finds TFIDF of the search query
def findSearchTFIDF(wordCloud, matrixdf, keywords):
    TFIDFList = []
    tsum = []
    for i in keywords:
        wordTFIDFdfList = findWordTFIDF(wordCloud, matrixdf, i)
        TFIDFList.append(wordTFIDFdfList)
    listTFIDF = pd.DataFrame(TFIDFList)
    tsum = listTFIDF.sum(axis=0)
    return tsum

#finds the recommendations based on the search query and dataset filename
def findRecommendations(filename, search):
    #generate word cloud
    sample = pd.read_csv(filename)
    titledcloud = generateWordCloudTitle(sample)
    abstractdcloud = generateWordCloud(sample)
    #generate word cloud matrix
    titleTFIDFMatrix = generateWordCloudMatrixTitle(sample)
    abstractTFIDFMatrix = generateWordCloudMatrix(sample)
    #fetch the search sentence
    #find keywords from the sentence
    sentence = findKeywords(search)
    #find TFIDF values of the keywords
    titlefinalTFIDF = findSearchTFIDF(titledcloud, titleTFIDFMatrix, sentence)
    abstractfinalTFIDF = findSearchTFIDF(abstractdcloud, abstractTFIDFMatrix, sentence)
    #calculate TFIDF Score
    if len(titlefinalTFIDF)>0:
        titlemaxVal = max(titlefinalTFIDF)
    else:
        titlemaxVal = 0
    if len(abstractfinalTFIDF)>0:
        abstractmaxVal = max(abstractfinalTFIDF)
    else:
        abstractmaxVal = 0
    if titlemaxVal==0:
        titlemulFrac = 0
    else:
        titlemulFrac = (1/titlemaxVal) * 75
    if abstractmaxVal==0:
        abstractmulFrac = 0
    else:
        abstractmulFrac = (1/abstractmaxVal) * 25
    titleTFIDFScore = [i * titlemulFrac for i in titlefinalTFIDF]
    abstractTFIDFScore = [i * abstractmulFrac for i in abstractfinalTFIDF]
    ts = [x + y for x, y in zip(titleTFIDFScore, abstractTFIDFScore)]
    fs = pd.DataFrame({'Final Score': ts})
    indexList = list(range(len(sample)))
    fscol = list(fs['Final Score'])
    pred = dict(zip(indexList, fscol))
    predSorted = sorted(pred, key=pred.get, reverse=True)
    return predSorted


# Influence score
def getHighlyInfluencedPapersScore(df):
    newCol = []
    max_value = max([float(i) for i in df["Highly Influenced Papers"].values])
    for ind in df.index:
        Highly_Influenced_Paper_Count = float(df['Highly Influenced Papers'][ind])
        Highly_Influenced_Paper_Score = ((Highly_Influenced_Paper_Count / max_value) * 100 * 0.4)
        newCol.append(Highly_Influenced_Paper_Score)
    return newCol


# Citation score
def getCitationScore(df):
    newCol = []
    my_lst = []
    max_value = max([float(i) for i in df["Citations"].values])
    for ind in df.index:
        citationCount = float(df['Citations'][ind])
        citationScore = ((citationCount / max_value) * 100 * 0.4)
        newCol.append(citationScore)
    return newCol


# Twitter mentions
def getTwitterMentionsScore(df):
    newCol = []
    max_value = max([float(i) for i in df["Twitter Mentions"].values])
    for ind in df.index:
        Twitter_Mentions_Count = float(df['Highly Influenced Papers'][ind])
        Twitter_Mentions_Score = ((Twitter_Mentions_Count / max_value) * 100 * 0.2)
        newCol.append(Twitter_Mentions_Score)
    # df2 = pd.DataFrame({'Twitter Mentions Score': newCol})
    return newCol


#rank records based on citationScore, InfluenceFactor, TwitterMentions
def rankRecommendations(predSorted, filename):
    dataset = pd.read_csv(filename)
    dataset['Citations'] = dataset['Citations'].str.strip().str.lower().str.replace(',', '')
    dataset['Highly Influenced Papers'] = dataset['Highly Influenced Papers'].str.strip().str.lower().str.replace(',', '')
    dataset['Twitter Mentions'] = dataset['Twitter Mentions'].str.strip().str.lower().str.replace(',', '')
    tmScore = getTwitterMentionsScore(dataset)
    citationScore = getCitationScore(dataset)
    ifScore = getHighlyInfluencedPapersScore(dataset)
    totalScore = [x + y + z for x, y, z in zip(ifScore, citationScore, tmScore)]
    finalScore = pd.DataFrame({'Final Score': totalScore})
    di = finalScore.iloc[predSorted].head(10)
    di = di.sort_values(by = 'Final Score', ascending =False)
    di['index'] = di.index
    recoIndexList = list(di['index'])
    finalrecodf = dataset.iloc[recoIndexList]
    return finalrecodf[['Title of the Article','Authors']]


def recommendForSearch(search):
    filename = 'final.csv'
    addKeywordsToDataSet(filename)
    predSorted = findRecommendations(filename, search)
    recommendations = rankRecommendations(predSorted, filename)
    return recommendations


# checks citation matches between base paper and dataset
def checkCitationMatch(baseCitList, index, df):
    potentialRecoList = []
    for i in range(len(df)):
        if i != index:
            citcols = df.iloc[i]['Reference Titles']
            citList = citcols.split(',')
            count = 0
            for citTitle in citList:
                if citTitle in baseCitList:
                    count = count + 1
            if count > 0:
                potentialRecoList.append(i)
    return potentialRecoList


# checks if any common paper exists for base paper and potential recommendable paper
def checkIfCommonPaperExists(index1, index2, title1, title2, df):
    flag = False
    for i in range(len(df)):
        if (i != index1 and i != index2):
            paper = df.iloc[[i]]
            citcol = list(paper['Reference Titles'])
            citList = citcol[0].split(',')
            if title1.iloc[0] and title2.iloc[0] in citList:
                flag = True
    return flag


# checks common papers for a base paper
def checkCommonPaper(index1, potentialRecoList, df):
    modifiedRecoList = []
    paper1 = df.iloc[[index1]]
    title1 = paper1['Title of the Article']
    for index2 in potentialRecoList:
        paper2 = df.iloc[[index2]]
        title2 = paper2['Title of the Article']
        flag = checkIfCommonPaperExists(index1, index2, title1, title2, df)
        if flag == True:
            modifiedRecoList.append(index2)
    return modifiedRecoList


# main function
def findMoreSimilar(index, filename):
    df = pd.read_csv(filename)
    result = []
    basePaperRecord = df.iloc[[index]]
    baseCitRecord = list(basePaperRecord['Reference Titles'])
    baseCitList = baseCitRecord[0].split(',')
    potentialRecoList = checkCitationMatch(baseCitList, index, df)
    finalRecoList = checkCommonPaper(index, potentialRecoList, df)
    if len(finalRecoList) == 0:
        result = finalRecoList
    else:
        for i in finalRecoList:
            result.append(df.iloc[i][['Title of the Article','Authors']])
    return result

@app.route('/home')
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/")
def search():
    return render_template("bootstrap_home.html")


@app.route("/process", methods=['GET'])
def process():
    logging.debug('Inside process, received request is  ' + str(request.args))
    query = request.args.get('query')
    result = recommendForSearch(query.lower())
    if(len(result != 0)):
        recommendation_table = create_recommendation_table(df=result)
        return render_template('table1.html', table=recommendation_table)
    else:
        return render_template('noMoreSimilar.html')


def create_recommendation_table(df):
    class TableCls(Table):
        classes = ['Table', 'table-striped', 'table-bordered', 'table-condensed']
        index = Col('Index',show=False)
        title = Col('Title of Paper')
        author = Col('Authors')
        edit = LinkCol('Show Similar', 'showSimilar', url_kwargs=dict(index='index'))

    items = []

    for index, row in df.iterrows():
        items.append(dict(index=index, title=row['Title of the Article'],author = row['Authors']))

    table = TableCls(items)
    return table

def create_similar_table(df):
    class TableClass(Table):
        classes = ['Table', 'table-striped', 'table-bordered', 'table-condensed']
        index = Col('Index',show=False)
        title = Col('Title of Paper')
        author = Col('Authors')

    items = []

    for index, row in df.iterrows():
        items.append(dict(index=index, title=row['Title of the Article'],author = row['Authors']))

    table = TableClass(items)
    return table


@app.route('/similar/<int:index>', methods=['GET'])
def showSimilar(index):
    filename = 'final.csv'
    result = findMoreSimilar(index, filename)
    if(len(result) != 0):
        dfObj = pd.DataFrame(result)
        table_html = create_similar_table(dfObj)
        return render_template('table.html', table=table_html)
    else:
        return render_template('noMoreSimilar2.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)