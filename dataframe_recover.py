import pandas as pd
import numpy as np 
import os
import modules.output_util as ou


args0 = "gen_list_1/"
args1 = "fgm0.009_temp_rows.npy"
file_path = "./out_dir/" + args0 + args1
	
rows = np.load(file_path)

print(rows)

f_out = ou.create_df_out(rows)
print(df_out)
path = "./out_dir/" + "gen_list_1" 
df_out.to_csv(path + "/" + csv_name + '_output_recover.csv', index=False)