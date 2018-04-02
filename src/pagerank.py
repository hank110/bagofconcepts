max_iteration=1000
epsilon=0.00001
damping_factor=0.15
output_name="pagerank.csv"


for concept, words in concept_to_words.items():
    # Initializer
    M_PageRank = defaultdict(int)
    total_coocc=0
    for word in words:
        M_PageRank[word]=sum(M_cooccurrence[word].values())
        total_coocc+=sum(M_cooccurrence[word].values())
    for word, pr_value in M_PageRank.items():
        M_PageRank[word]=pr_value/total_coocc
        #M_PageRank[word]=pr_value//(get_tf(word)/get_df(word))
    
    # Iteration
    for _ in range(max_iteration):
        old_PageRank=M_PageRank
        old_PageRank_vector=np.array(list(old_PageRank.values()))
        for word, pr_value in old_PageRank.items():
            update_pr_value=0
            for linked_word in M_cooccurrence[word].keys():
                #update_pr_value+=(M_cooccurrence[word][linked_word]*old_PageRank[linked_word])/sum(M_cooccurrence[linked_word].values())
                update_pr_value+=old_PageRank[linked_word]/len(M_cooccurrence[linked_word].keys())
            ## Should be divided by the number of documents
            if (len(M_cooccurrence[word].keys())==0): alpha=0
            else: alpha = 1.0 / float(len(M_cooccurrence[word].keys())+0.0000001) * damping_factor
            #alpha = 1.0 / float(len(words)+0.0000001) * damping_factor
            M_PageRank[word]=update_pr_value*(1-damping_factor)+alpha
        delta=math.sqrt(np.sum(np.power(np.array((list(M_PageRank.values()))-old_PageRank_vector),2)))
        if delta < epsilon:
            print("...PageRank converged after %s iterations" %str(_))
            break
        if _ % 100 == 0: print("...number of iterations: %s" %str(_))
    print("PageRank calculation completed")
    print("delta: %s" %str(delta))
    for word, pr_value in sorted(M_PageRank.items(), key=lambda x: x[1], reverse=True):
        with open(output_name, "a") as f:
            f.write('%d, %s, %.5f\n' % (concept, word, pr_value))
