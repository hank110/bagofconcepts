import boc

a=boc.BOCModel(doc_path="../sample/sample_data/sample_articles.txt")
mtx,w2c,idx=a.fit(save_path="./")

print(mtx.shape)
print(w2c[203])
print(idx[20])
