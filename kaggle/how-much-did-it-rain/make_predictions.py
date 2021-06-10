import pandas as pd
import numpy as np
import functions as fn

model_path = '../models/'
path = '../processed/'
sfx = '_final_subm'


predictions1 = fn.make_prediction1(fn.load_xgb_model(model_path+'bst1_1'+sfx),path)
predictions2 = fn.make_prediction2(fn.load_xgb_model(model_path+'bst2_1'+sfx),path)
predictions3 = fn.make_prediction3(fn.load_xgb_model(model_path+'bst3_1'+sfx),path)

#NOTE set use_xtra_features to False as part of a faster solution
predictions4 = fn.make_prediction4(fn.load_xgb_model(model_path+'bst4_1'+sfx),path,use_xtra_features=True) #change to False for speed up
predictions5 = fn.make_prediction5(fn.load_xgb_model(model_path+'bst5_1'+sfx),path,use_xtra_features=True) #change to False for speed up


#This is the post processing, the more large values in a subset the more you can subtract
predictions1 = predictions1 
predictions2 = predictions2 - 0.0025 
predictions3 = predictions3 - 0.0024 
predictions4 = predictions4 - 0.0052 
predictions5 = predictions5 - 0.0084 


predictions = pd.concat([predictions1,predictions2,predictions3,predictions4,predictions5])
predictions = predictions.sort_index()

subm_columns = ['Predicted'+str(x) for x in range(70)]
predictions.columns = subm_columns
predictions.index.name = 'Id'

predictions.to_csv('../output/final_subm_pp.csv',float_format='%.4f')
subm_new = pd.read_csv('../output/final_subm_pp.csv',index_col=0)

#checks that the solution meets the requirements to be properly scored. 
print fn.check_solution(subm_new.values)

