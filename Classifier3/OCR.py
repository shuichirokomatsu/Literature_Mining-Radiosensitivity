import os, csv, time, re, codecs
from collections import Counter
import nltk

listFile = 'filelist.csv'
dateFormat = '%Y/%m/%d %H:%M:%S'
    
csvFile = open(listFile, 'w', newline='')
csvWriter = csv.writer(csvFile)
header = ['PMID', 'sf1', 'sf2', 'sf3', 'sf4', 'sf5', 'sf6', 'sf7', 'sf8', 'sf9', 'sf10', 'sf11', 'sf12', 'sf13', 'sf14', 'sf15', 'sf16', 'sf17', 'sf18',  'sf19', 'sf20', 'd10', 'd50']
csvWriter.writerow(header)

for file in os.listdir():
    row = []
    target_text = file
    row.append(os.path.basename(file))
    f = codecs.open(file, 'r', 'utf-8', 'ignore')
    data = f.read()
    words = re.split(r'\s|\,|\.|\(|\)', data.lower())
    counter = Counter(words)
    tokens = nltk.word_tokenize(data)
    tokens_l = [w.lower() for w in tokens]
    bigrams = nltk.bigrams(tokens_l)
    fd = nltk.FreqDist(bigrams)
    row.append(counter['sf1'])
    row.append(counter['sf2'])
    row.append(counter['sf3'])
    row.append(counter['sf4'])
    row.append(counter['sf5'])
    row.append(counter['sf6'])
    row.append(counter['sf7'])
    row.append(counter['sf8'])
    row.append(counter['sf9'])
    row.append(counter['sf10'])
    row.append(counter['sf11'])
    row.append(counter['sf12'])
    row.append(counter['sf13'])
    row.append(counter['sf14'])
    row.append(counter['sf15'])
    row.append(counter['sf16'])
    row.append(counter['sf17'])
    row.append(counter['sf18'])
    row.append(counter['sf19'])
    row.append(counter['sf20'])
    row.append(counter['d10'])
    row.append(counter['d50'])

    csvWriter.writerow(row)

csvFile.close()