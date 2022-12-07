# helper tools for summarizing data
import numpy as np

from meta_utils.plot_format import get_mod_id_order, display_names

def best_block_table(all_data, mod_ids, a_m, higher_better=True):
	# sort model id ordering
    all_data_dict = {}
    for i in range(len(mod_ids)):
        all_data_dict[mod_ids[i]] = all_data[i]
    id_ord = get_mod_id_order(mod_ids)
    # make table
    print('============================================================')
    print(a_m)
    print('============================================================')
    print('%s%s%s'%('model_id'.ljust(30),'best blk'.ljust(20),'best score'))
    for mod_id in id_ord:
    	data = all_data_dict[mod_id]
    	if higher_better:
    		best_blk = np.argmax(data)
    	else:
    		best_blk = np.argmin(data)
    	best_res = data[best_blk]
    	disp_id = display_names(mod_id)
    	best_blk_str = '%i (of %i)'%(best_blk+1, data.shape[0])
    	print('%s%s%.3f'%(disp_id.ljust(30),best_blk_str.ljust(20),best_res))
    print('')
