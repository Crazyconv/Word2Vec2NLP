# from sentences import Sentences

# sentences = Sentences("/Users/Crazyconv/Conv/DEVELOPMENT/GitFolder/Word2Vec2NLP/dataset/train")
# i = 0
# j = 5
# for line in sentences.paragraph_iterator():
#     i += 1

# print i

# from util import *
# process_option = ProcessOption()
# print str(process_option)

# from multiprocessing import Manager, Process
# import multiprocessing as mp
# import numpy as np

# def process(wordset, docs, index):
#     doc_vector = np.frombuffer(dv).reshape((3,3))
#     for word in docs[index]:
#         if word in wordset:
#             doc_vector[index] += wordset[word]


# if __name__ == '__main__':
#     doc_vector = np.zeros((9,1), dtype="float32")
#     wordset = {"1": np.array([1,2,3]), "2": np.array([2,3,4])}
#     docs = [["1", "2"], ["3"], ["2", "3"]]

#     manager = Manager()
#     ws = manager.dict(wordset)
#     ds = manager.list(docs)

#     # p = [0,0,0]
#     # for i in range(3):
#     #     p[i] = Process(target=process, args=(ws, ds, dv, i))
#     #     p[i].start()

#     # for i in range(3):
#     #     p[i].join()

#     dv = mp.Array('d', doc_vector, lock=False)
#     print np.frombuffer(dv)
#     # print dv
#     pool = mp.Pool(3)
    
#     for i in range(3):
#         pool.apply_async(process, [ws, ds, i,])

#     pool.close()
#     pool.join()

#     # print doc_vector
#     print np.frombuffer(dv).reshape((3, 3))


from multiprocessing import Manager, Process
import multiprocessing as mp
import numpy as np
import mputil

def process(wordset, docs, index):
    doc_vector = np.frombuffer(mputil.toShare).reshape((3,3))
    for word in docs[index]:
        if word in wordset:
            doc_vector[index] += wordset[word]

def initprocess(share):
    mputil.toShare = share

def main():
    doc_vector = np.zeros((9,1), dtype="float32")
    wordset = {"1": np.array([1,2,3]), "2": np.array([2,3,4])}
    docs = [["1", "2"], ["3"], ["2", "3"]]

    manager = Manager()
    ws = manager.dict(wordset)
    ds = manager.list(docs)

    # p = [0,0,0]
    # for i in range(3):
    #     p[i] = Process(target=process, args=(ws, ds, dv, i))
    #     p[i].start()

    # for i in range(3):
    #     p[i].join()

    dv = mp.Array('d', doc_vector, lock=False)
    print np.frombuffer(dv).reshape((3, 3))
    pool = mp.Pool(initializer=initprocess, initargs=[dv])
    
    for i in range(3):
        pool.apply_async(process, [ws, ds, i,])

    pool.close()
    pool.join()

    # print doc_vector
    print np.frombuffer(dv).reshape((3, 3))

if __name__ == '__main__':
    main()