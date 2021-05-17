######################################################
# author: Devin Anzelmo, devinanzelmo@gmail.com
# licence: FreeBSD

"""
Copyright (c) 2015, Devin Anzelmo
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 OF THE POSSIBILITY OF SUCH DAMAGE.

"""


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
    train, integer_labels,actual_labels, cutoff = fn.load_train_data(train_path,1,4,offset_amount)

    reduced_labels = fn.aggregate_labels([[range(3,7),3],[range(7,70),4]], integer_labels).iloc[:,0] #.iloc becuase series and df don't behave the same

    data = (train.iloc[cutoff:,:],reduced_labels.iloc[cutoff:],train.iloc[:cutoff,:],reduced_labels.iloc[:cutoff])

    bst1 = fn.train_tree_xgb(data,0.015, 1.5, 9, 45, .55, .55,5, num_threads, num_over)
    bst1.save_model(model_path+'bst2_1'+model_name_suffix)

if __name__ == "__main__":
    main()
