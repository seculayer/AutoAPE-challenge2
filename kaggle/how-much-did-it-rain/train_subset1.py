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
    train, integer_labels,actual_labels, cutoff = fn.load_train_data(train_path,1,1,offset_amount)

    #drop the columns with constant values  
    train = train.loc[:,train.mean() != -99999]

    #aggregate the original labels into 3 groups, 0mm,1mm, and 2-69mm
    reduced_labels = fn.aggregate_labels([[range(2,70),2]], integer_labels).iloc[:,0] #.iloc becuase series and df don't behave the same
    
    #split into a train and validation set for early stopping, this makes the call to xgb readable
    data = (train.iloc[cutoff:,:],reduced_labels.iloc[cutoff:],train.iloc[:cutoff,:],reduced_labels.iloc[:cutoff])

    
    #train_tree_xgb(data,eta, gamma, max_d, min_child, subsamp, col_samp,num_classes, num_threads, num_over=3,eval_func=None):
    bst1 = fn.train_tree_xgb(data, 0.015, 1.5, 9, 55, .45, .55,3, num_threads, num_over)

    #done with this model save for later when we make the predictions
    bst1.save_model(model_path+'bst1_1'+model_name_suffix)


if __name__ == "__main__":
    main()
