import csv
import re
from topia.termextract import extract

extractor = extract.TermExtractor()

vocabulary=[]
with open('C:\Minor Project\Kaggle Dataset\jobs.tsv', 'rb') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    titles=[]
    for row in tsv_reader:
    	print row
    	titles=row
    	break
k=0
while k < len(titles) :
	vocabulary.append(set())
	k+=1

print vocabulary

count=0

with open('C:\Minor Project\Kaggle Dataset\jobs.tsv', 'rb') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    titles=[]
    for row in tsv_reader:
    	pattern = re.compile(r"<.>")
    	row= [pattern.sub("", item) for item in row]
    	pattern = re.compile(r"</.>")
    	row= [pattern.sub("", item) for item in row]
    	pattern = re.compile(r"<a.*>")
    	row= [pattern.sub("", item) for item in row]
    	pattern = re.compile(r"&nbsp;")
    	row= [pattern.sub("", item) for item in row]
    	pattern = re.compile(r"\\r")
    	row= [pattern.sub(" ", item) for item in row]
    	row= [item.lower() for item in row]
    	pattern = re.compile(r" +")
    	row= [pattern.sub(" ", item) for item in row]
    	pattern = re.compile(r"\\n")
    	row= [pattern.sub(" ", item) for item in row]
    	pattern = re.compile(r" \*")
    	row= [pattern.sub(" ", item) for item in row]
    	print row
    	extractor.filter = extract.permissiveFilter
    	l=0
    	while l < len(row):
    		#vocabulary[l].append()
    		print "terms:-"
    		for term in extractor(row[l]):
    			vocabulary[l].add(term[0])
    		print vocabulary[l]
    		l+=1

    	count+=1
    	if count==50:
    		break
print vocabulary

data=[]

count=0
with open('C:\Minor Project\Kaggle Dataset\jobs.tsv', 'rb') as tsv_file:
	tsv_reader = csv.reader(tsv_file, delimiter='\t')
	for row in tsv_reader:
		if count==50:
			break
		pattern = re.compile(r"<.>")
		row= [pattern.sub("", item) for item in row]
		pattern = re.compile(r"</.>")
		row= [pattern.sub("", item) for item in row]
		pattern = re.compile(r"<a.*>")
		row= [pattern.sub("", item) for item in row]
		pattern = re.compile(r"&nbsp;")
		row= [pattern.sub("", item) for item in row]
		pattern = re.compile(r"\\r")
		row= [pattern.sub(" ", item) for item in row]
		row= [item.lower() for item in row]
		pattern = re.compile(r" +")
		row= [pattern.sub(" ", item) for item in row]
		pattern = re.compile(r"\\n")
		row= [pattern.sub(" ", item) for item in row]
		pattern = re.compile(r" \*")
		row= [pattern.sub(" ", item) for item in row]
		fvector=[]
		l=0
		while l < len(row):
			vector=[]
			for vocab in vocabulary[l]:
				if vocab in row[l]:
					vector.append(1)
				else:
					vector.append(0)	
			l+=1
			fvector.append(vector)
		#print fvector
		data.append(fvector)
		count+=1
	f = open('feature_vectors.txt', 'w')	
	print "feature vectors!!:-"
	for da in data:
		print da
		print>> f,da
		print>>f,"\n"
