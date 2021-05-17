import pandas as pd
import numpy as np
import random
import functions as fn 

def main():
    random.seed(11)
    np.random.seed(11)
    
    train_path = '../processed/'
    model_path = '../models/'

    model_name_suffix = '_final_subm'
    offset_amount = 0.07# fraction of the train set to use as hold out
    num_over = 2
    num_threads  = 7

    #generate trainset, labels, and test set based on the number of valid radar readings in the original dataset. 
    train, integer_labels,actual_labels, cutoff = fn.load_train_data(train_path,3,8,offset_amount)

    reduced_labels = fn.aggregate_labels([[range(3,7),3],[range(7,70),4]], integer_labels).iloc[:,0] #.iloc becuase series and df don't behave the same

    data = (train.iloc[cutoff:,:],reduced_labels.iloc[cutoff:],train.iloc[:cutoff,:],reduced_labels.iloc[:cutoff])

    bst1 = fn.train_tree_xgb(data, 0.02, 1.5, 14, 45, .45, .5,5, num_threads, num_over)
    bst1.save_model(model_path+'bst3_1'+model_name_suffix)


if __name__ == "__main__":
    main()
