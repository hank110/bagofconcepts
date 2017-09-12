# Word2Vec Training Setup
## input: training document (Need to arrange the doc such that each line = one sentence in txt)
## parameters: embedding dimension, window size, minimum word frequency threshold

document='/home/hank/Backup_data/data/r52-test-all-terms.txt'
dimensions=[300]	
context=6				
min_freq=20
num_concepts=[70]
w2v_model=[] 						
