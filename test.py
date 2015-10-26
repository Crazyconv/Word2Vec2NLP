from sentences import Sentences

sentences = Sentences("/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset")
i = 0
j = 5
for line in sentences:
	print line
	print
	i += 1
	if(i >= j):
		break