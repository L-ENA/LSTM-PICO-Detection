from gensim.models.keyedvectors import KeyedVectors




def makeTxt():
    """Procedure to build text file from binary embedding data


    Args:
        none

    """
    model = KeyedVectors.load_word2vec_format('/home/lena/Dokumente/Master/dissertation/Data/wikipedia-pubmed-and-PMC-w2v.bin', binary=True, limit = 20)
    model.save_word2vec_format('/home/lena/Dokumente/Master/dissertation/Data/embMini.txt', binary=False)
    print('done creating text files')


file = open('testfile.txt', 'w')
file.write('Why? Because we can.')

file.close()

makeTxt()



#if __name__ == "__makeTxt__":
    #main()
