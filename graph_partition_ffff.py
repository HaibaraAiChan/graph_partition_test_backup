import numpy 
import dgl
from numpy.core.numeric import Infinity
import multiprocessing as mp
import torch
import time
from statistics import mean


maxDegree=0
gains={}
bit_dict={}
side=0
block_to_graph=None

# locked_nodes={};

def InitializeBitList(p_list):
    global bit_dict
    nums_p=len(p_list); 
    bit_dict={}
    for i in range (nums_p-1):
        A = p_list[i]
        B = p_list[i + 1]; 
        for k in A:
            bit_dict[k] = 0
        for m in B:
            bit_dict[m] = 1
        
    return 
    
def redundancy_check(cur_part_in_size,ideal_part_in_size,target_redun):
    redundancy_flag=True
    r = cur_part_in_size/ideal_part_in_size
    if r > target_redun:
        redundancy_flag=False
    return redundancy_flag
    
def balance_check_2(A_in,B_in,alpha):
    balance_flag=True
    avg = (len(A_in)+len(B_in))/2
    if abs(len(B_in)-len(A_in)) >alpha*avg:
        balance_flag=False
    return balance_flag
    

    
def getRedundancyRate(len_A, len_B, ideal_part_in_size):
    rate =0
    ratio_A = len_A/ideal_part_in_size
    ratio_B = len_B/ideal_part_in_size
    # cost = max(ratio_A,ratio_B)
    rate = mean([ratio_A,ratio_B])
    return rate, ratio_A,ratio_B   #   minimize the max ratio of 
    
def get_mini_batch_size(full_len,num_batch):
	mini_batch=int(full_len/num_batch)
	if full_len%num_batch>0:
		mini_batch+=1
	# print('current mini batch size of output nodes ', mini_batch)
	return mini_batch
	
def gen_batch_output_list(OUTPUT_NID,indices,mini_batch):
	
	map_output_list = list(numpy.array(OUTPUT_NID)[indices])
		
	batches_nid_list = [map_output_list[i:i + mini_batch] for i in range(0, len(map_output_list), mini_batch)]
			
	output_num = len(OUTPUT_NID)
	
	# print(batches_nid_list)
	weights_list = []
	for i in batches_nid_list:
		# temp = len(i)/output_num
		weights_list.append(len(i)/output_num)
		
	return batches_nid_list, weights_list
	
	
def random_shuffle(full_len):
	indices = numpy.arange(full_len)
	numpy.random.shuffle(indices)
	return indices

def gen_batched_seeds_list(OUTPUT_NID, args):
	'''

	Parameters
	----------
	OUTPUT_NID: final layer output nodes id (tensor)
	args : all given parameters collection

	Returns
	-------

	'''
	selection_method = args.selection_method
	num_batch = args.num_batch
	# mini_batch = 0
	full_len = len(OUTPUT_NID)  # get the total number of output nodes
	mini_batch_size=get_mini_batch_size(full_len,num_batch)
	args.batch_size=mini_batch_size
	if selection_method == 'range_init_graph_partition' :
		indices = [i for i in range(full_len)]
	
	if selection_method == 'random_init_graph_partition' :
		indices = random_shuffle(full_len)
		
	elif selection_method == 'similarity_init_graph_partition':
		indices = torch.tensor(range(full_len)) #----------------------------TO DO
	
	batches_nid_list, weights_list=gen_batch_output_list(OUTPUT_NID,indices,mini_batch_size)
	
	return batches_nid_list, weights_list


    




def get_weight_list(batched_seeds_list):
    
    output_num = len(sum(batched_seeds_list,[]))
    # print(output_num)
    weights_list = []
    for seeds in batched_seeds_list:
		# temp = len(i)/output_num
        weights_list.append(len(seeds)/output_num)
    return weights_list



def get_p_nid(nid, p_id):
    global bit_dict
    
    if bit_dict[nid]==p_id:
        return nid
    return


def exists(it):
    return (it is not None)

def get_two_partition_seeds():
    global bit_dict
    
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    
    return  A_o, B_o
    


def balance_checking(alpha): 
    global bit_dict
    flag=False
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    print('\t\t\t length of A_o,B_o  '+str(len(A_o))+' '+str(len(B_o)))
          
    
    len_A_part,len_B_part = get_src_nodes_len(A_o,B_o)
    len_=len_A_part+len_B_part
    avg = len_/2
    if len_B_part>0 and len_A_part>0 and abs(len_A_part-len_B_part) < avg*alpha:
        flag=True
    return flag, len_A_part,len_B_part  

def move_group_nids_balance_redundancy_check(bit_dict_origin, side, nids,alpha, red_rate,ideal_part_in_size):
    for nid in nids:
        bit_dict_origin[nid]=1-side
        
    A_o=[k for k in bit_dict_origin if bit_dict_origin[k] == 0]
    B_o=[k for k in bit_dict_origin if bit_dict_origin[k] == 1]
    
    for nid in nids:
        bit_dict_origin[nid]=1-bit_dict_origin[nid]
        
    balance_flag=False
   
    # t1=time.time()
    len_A_part,len_B_part = get_src_nodes_len(A_o,B_o)
    # print('get_src_nodes_len ', time.time()-t1)
    len_=len_A_part+len_B_part
    avg = len_/2
    
    if len_B_part>0 and len_A_part>0 and abs(len_A_part-len_B_part) < avg*alpha:
        balance_flag=True
    else:
        balance_flag=False
    # now test -----------------------------------8---------------88-8---8-8-888-8-8-888
    # balance_flag=True    # now test -----------------------------------8---------------88-8---8-8-888-8-8-888
    # now test -----------------------------------8---------------88-8---8-8-888-8-8-888
    red, ratio_A, ratio_B=getRedundancyRate(len_A_part, len_B_part, ideal_part_in_size)
    
    red_flag=True if red<red_rate else False
    if red_flag and balance_flag:
        print(' redundancy '+str(red))
    return balance_flag and red_flag
 

def calculate_redundancy_A_o(idx,i, A_o, B_o,locked_nodes):
    global block_to_graph
    
    gain=0
    in_nids=block_to_graph.predecessors(i).tolist()
    # A_src,B_src=get_src_nodes(A_o,B_o)
    gain_pos=len(list(set(in_nids).intersection(set(B_o))))
    gain_neg=len(list(set(in_nids).intersection(set(A_o)))) 
    
    gain=gain_pos-gain_neg 
    if gain>=0 and  not locked_nodes[i] :
        return (idx,i)
    
    return (idx,None)


def calculate_redundancy_B_o(idx,i, A_o, B_o,locked_nodes):
    global block_to_graph
    
    gain=0
    in_nids=block_to_graph.predecessors(i).tolist()
    # A_src,B_src=get_src_nodes(A_o,B_o)
    gain_pos=len(list(set(in_nids).intersection(set(A_o))))
    gain_neg=len(list(set(in_nids).intersection(set(B_o)))) 
    
    gain=gain_pos-gain_neg 
    if gain>=0 and not locked_nodes[i] :
        return (idx,i)
    
    return (idx,None)
                

def updateRed_group(locked_nodes, cur_rate, alpha,ideal_part_in_size):
    
    global  bit_dict
    #------------------------
    
    bit_dict=balance_check_and_exchange(bit_dict, alpha)
    
    #-------------------------
    nid_to_move=[]
    
    A_o,B_o=get_two_partition_seeds()
    
    if len(A_o)>1: 
        pool = mp.Pool(mp.cpu_count())
        tmp_gains = pool.starmap_async(calculate_redundancy_A_o, [(idx, i, A_o, B_o, locked_nodes) for idx, i in enumerate(A_o)]).get()
        pool.close()
        
        nid_to_move = [list(r)[1] for r in tmp_gains]
        nid_to_move=list(filter(lambda v: v is not None, nid_to_move))
        # print(nid_to_move)
        
        # print('the number of node ready to move A-> B is :', len(nid_to_move))
        print()
        
        while len(nid_to_move)>0:    
            if not move_group_nids_balance_redundancy_check(bit_dict, 0, nid_to_move,alpha, cur_rate,ideal_part_in_size):
                # print('error move_group_nids_balance_redundancy_check !!!!!!!!!!!!!! B_o')
                
                mid=int(len(nid_to_move)/2)
                nid_to_move=nid_to_move[:mid]
            
            else:
                if len(A_o)==len(nid_to_move):
                    nid_to_move=nid_to_move[:(len(nid_to_move)-1)]
                    
                break
        print('the number of node really move A-> B is :', len(nid_to_move))
        for nid in nid_to_move:
            bit_dict[nid]=1
                    
        for i in nid_to_move:
            locked_nodes[i]=True
        # gains=initializeBucketSort(i,gain, 0) # 0: left side bucket
        # print('rate of zero and positive gains: ', len(nid_to_move)/idx)
     #---------------------------------------------------------------------------------------------
    if len(B_o)>1: 
        ready_to_move=[]
        
        pool = mp.Pool(mp.cpu_count())
        tmp_gains = pool.starmap_async(calculate_redundancy_B_o, [(idx, i, A_o, B_o, locked_nodes) for idx, i in enumerate(B_o)]).get()
        pool.close()
        # print(tmp_gains)
        ready_to_move = [list(r)[1] for r in tmp_gains]
        ready_to_move=list(filter(lambda v: v is not None, ready_to_move))
        # print(ready_to_move)

        # print('the number of node ready to move B->A is :', len(ready_to_move))
        while len(ready_to_move)>0:    
            if not move_group_nids_balance_redundancy_check(bit_dict, 1, ready_to_move,alpha, cur_rate,ideal_part_in_size):
                # print('error move_group_nids_balance_redundancy_check !!!!!!!!!!!!!! B_o')
                mid=int(len(ready_to_move)/2)
                ready_to_move=ready_to_move[:mid]
            else:
                if len(B_o)==len(ready_to_move):
                    ready_to_move=ready_to_move[:(len(ready_to_move)-1)]
                break
            # print('rate of zero positive: ', len(ready_to_move)/idx)
        print('the number of node really  move  from B->A is :', len(ready_to_move))
        for i in ready_to_move:
            locked_nodes[i]=True
        for nid in ready_to_move:
            bit_dict[nid]=0
        
    return    locked_nodes             
                 
                
                         
def balance_check_all_partitions(partition_dst_src_list,alpha):
    balance_flag = True
    for i in range(len(partition_dst_src_list)-1):
        A_nids= partition_dst_src_list[i]
        B_nids= partition_dst_src_list[i+1]
        avg = (len(A_nids)+len(B_nids))/2
        if abs(len(B_nids)-len(A_nids)) >alpha*avg:
            balance_flag=False
            return balance_flag
    return balance_flag
    
    
    
    
def exchange_bit_dict_side(bit_dict):
    bit_dict={i: 1-bit_dict[i] for i in bit_dict}
    return bit_dict





def balance_check_and_exchange(bit_dict,alpha):
    
    A_o=[k for k in bit_dict if bit_dict[k] == 0]
    B_o=[k for k in bit_dict if bit_dict[k] == 1]
    
    len_A_part,len_B_part = get_src_nodes_len(A_o,B_o)
    # if left partition size is less than right partition size, exchange side;
    # otherwise, do not exchange
    if len_B_part>0 and len_A_part>0 :
        if len_A_part-len_B_part < 0:
            bit_dict={i: 1-bit_dict[i] for i in bit_dict}
            
        # if len_A_part>len_B_part:
        #     k= A_o[0]
        #     if bit_dict[k]!=0: # if bit_dict[ A_o[k] ] == 1: exchange side
        #         bit_dict={i: 1-bit_dict[i] for i in bit_dict}
                
            # for k in A_o:
            #     bit_dict[k] = 0
            # for m in B_o:
            #     bit_dict[m] = 1
                
        # if len_A_part<len_B_part:
        #     k= B_o[0]
        #     if bit_dict[k]!=0: # if bit_dict[ B_o[k] ] == 1: exchange side
        #         bit_dict={i: 1-bit_dict[i] for i in bit_dict}
                                # otherwise keep original side
            # for k in A_o:
            #     bit_dict[k] = 1
            # for m in B_o:
            #     bit_dict[m] = 0
   
    return bit_dict
    # return bit_dict, side
    

    
def get_partition_src_list_len(batched_seeds_list,ideal_part_in_size):
    global block_to_graph
    
    partition_src_list_len=[]
    redundancy_list=[]
    for seeds_nids in batched_seeds_list:
        in_nids = get_in_nodes(seeds_nids)
        part=list(set(seeds_nids+in_nids))
        partition_src_list_len.append(len(part))
        redundancy_list.append(len(part)/ideal_part_in_size)
    return partition_src_list_len,redundancy_list


# def ___get_frontier(idx, seeds):
#     global block_to_graph
    
#     frontier=dgl.in_subgraph(block_to_graph, seeds)
#     src=set(list(frontier.edges())[0].tolist())
    
#     return idx,list(src)
    
def get_src_nodes(seeds_1,seeds_2):
       
    global block_to_graph
    
    # tt1=time.time()
    
    in_ids_1=list(block_to_graph.in_edges(seeds_1))[0].tolist()
    src_1= list(set(in_ids_1+seeds_1))
    in_ids_2=list(block_to_graph.in_edges(seeds_2))[0].tolist()
    src_2= list(set(in_ids_2+seeds_2))
    # frontier_1=dgl.in_subgraph(block_to_graph, seeds_1)
    # src_1=set(list(frontier_1.edges())[0].tolist())
    
    # frontier_2=dgl.in_subgraph(block_to_graph, seeds_2)
    # src_2=set(list(frontier_2.edges())[0].tolist())
    
    # print('time for frontier in subgraph : ', time.time()-tt1 )
    return src_1, src_2

def get_src_nodes_len(seeds_1,seeds_2):
   
    global block_to_graph
    # t0=time.time()
    in_ids_1=list(block_to_graph.in_edges(seeds_1))[0].tolist()
    src_len_1= len(list(set(in_ids_1+seeds_1)))
    in_ids_2=list(block_to_graph.in_edges(seeds_2))[0].tolist()
    src_len_2= len(list(set(in_ids_2+seeds_2)))
    
    # frontier_1=dgl.in_subgraph(block_to_graph, seeds_1)
    # src_len_1=len(set(list(frontier_1.edges())[0].tolist()))
    # # print('frontier_1',src_len_1)
    # frontier_2=dgl.in_subgraph(block_to_graph, seeds_2)
    # src_len_2=len(set(list(frontier_2.edges())[0].tolist()))
    return src_len_1, src_len_2


def get_in_nodes(seeds):
    global block_to_graph
    in_ids=list(block_to_graph.in_edges(seeds))[0].tolist()
    in_ids= list(set(in_ids))
    return in_ids
    
def update_Batched_Seeds_list(batched_seeds_list, bit_dict,i, j):

    batch_i=[k for k in bit_dict if bit_dict[k]==0]
    batch_j=[k for k in bit_dict if bit_dict[k]==1]`
    if set(batch_i)!= set(batched_seeds_list[i]):
        batched_seeds_list.remove(batched_seeds_list[i])
        batched_seeds_list.insert(i,batch_i)
    if set(batch_j)!= set(batched_seeds_list[j]):
        batched_seeds_list.remove(batched_seeds_list[j])
        batched_seeds_list.insert(j,batch_j)

    return batched_seeds_list
    
# def update_Batched_Seeds_list_final(batched_seeds_list, bit_dict,i, j):
    
#     batch_i=[k for k in bit_dict if bit_dict[k]==0]
#     batch_j=[k for k in bit_dict if bit_dict[k]==1]
#     batched_seeds_list=batched_seeds_list[:-2]
#     batched_seeds_list.append(batch_i)
#     batched_seeds_list.append(batch_j)
    
#     return batched_seeds_list

# def gen_random_batched_seeds_list(batched_seeds_list):
#     num_batch=len(batched_seeds_list)
#     output_nids= sum(batched_seeds_list,[])
#     full_len=len(output_nids)
#     indices = random_shuffle(full_len)
#     mini_batch_size=get_mini_batch_size(full_len,num_batch)
#     batches_nid_list, weights_list=gen_batch_output_list(output_nids,indices,mini_batch_size)
	
    # return batches_nid_list

def print_len_of_batched_seeds_list(batched_seeds_list):
    batch_list_str=''
    len_=''
    for i in batched_seeds_list:
        batch_list_str=batch_list_str+str(i)+', '
        len_=len_+str(len(i))+', '
    # batch_list_str=' '.join(batched_seeds_list)
    print('batch_seeds_list len ')
    # print(batch_list_str) 
    print(len_)
    
def print_len_of_partition_list(partition_src_list_len):
    # batch_list_str=''
    len_=''
    for i in partition_src_list_len:
        # batch_list_str=batch_list_str+str(i)+', '
        
        len_=len_+str(i)+', '
    # batch_list_str=' '.join(batched_seeds_list)
    print('partition_src_list len ')
    # print(batch_list_str) 
    print(len_)
    

def walk_terminate_1(g,red_rate, args,ideal_part_in_size):
    global bit_dict
    # global side
    redundancy_tolarent_steps=args.redundancy_tolarent_steps
    
    bestRate =red_rate
    best_bit_dict=bit_dict
    print('\twalk terminate 1')
    t1=time.time()
    bit_dict=balance_check_and_exchange(bit_dict, args.alpha) 
    # if left partition size is smaller than right partition, exchange them
    # make sure the left partition size is larger or equal with right partition size.
    # then, we can only focus the first step move nodes from the larger partition to the smaller one.
    A_o,B_o= get_two_partition_seeds() # get A_o, B_o based on global variable bit_dict
    subgraph_o = A_o+B_o
    locked_nodes={id:False for id in subgraph_o}
    # steps_=redundancy_tolarent_steps
    steps_=2
    t_b=time.time()
    for i in range(steps_):
        # tt=time.time()
        locked_nodes=updateRed_group(locked_nodes,red_rate,args.alpha,ideal_part_in_size)
        # tt_e=time.time()
        # if i % 1==0:
        #     print('\t\t\tone update redundancy spend time _*_\t', tt_e-tt)	
        #     print('\t\t  --------------------------------------------walk terminate 1 step ' ,i)
        
        balance_flag, len_A_part,len_B_part = balance_checking(args.alpha)
        print('\t\t\t len_A_part, len_B_part: (', str(len_A_part)+', '+ str(len_B_part)+')')
        
        tmpRate,ratio_A,ratio_B =getRedundancyRate(len_A_part,len_B_part,ideal_part_in_size)
        print('\t\t\t redundancy rate (ration_mean, ratio_A, ratio_B): '+str(tmpRate)+' '+str(ratio_A)+' '+ str(ratio_B))
        
        if balance_flag:
            print('\t\t\t ---balanced                  ----*---*----redundancy rate: ', tmpRate)
            if tmpRate < bestRate: 
                bestRate = tmpRate
                best_bit_dict = bit_dict
        else: # if partition A and B are not balanced, ex
            print()
            print(' it is not balance after updateRed_group')
            print()
            bit_dict=exchange_bit_dict_side(bit_dict) 
            
        
    t_e=time.time()
    print('\t\tupdate redundancy of segment  spend: ', t_e-t_b)
    if (bestRate < red_rate) : #is there improvement? Yes
        bit_dict = best_bit_dict
        print(' there are improvement, the best redundancy rate this terminate is ', bestRate)
        return True, bestRate
    
    return False, bestRate
    

def graph_partition_variant( batched_seeds_list, block_2_graph, args):
    
    global maxDegree
    global gains
    global bit_dict
    global block_to_graph
    print('----------------------------graph partition start---------------------')
    
    full_batch_graph_nids_size=len(block_2_graph.srcdata['_ID'])
    ideal_part_in_size=(full_batch_graph_nids_size/args.num_batch)
    full_batch_seeds = block_2_graph.dstdata['_ID'].tolist()
    num_batch=args.num_batch
    balance_flag = False
    
    # print(list(block_2_graph.edges()))
    src_ids=list(block_2_graph.edges())[0]
    dst_ids=list(block_2_graph.edges())[1]
    g = dgl.graph((src_ids, dst_ids))
    g=dgl.remove_self_loop(g)
    # from draw_graph import draw_graph
    # draw_graph(g)

    block_to_graph = g # set g to the global variable: block to graph
    # global locked_nodes
    print('{}-'*40)
    print()

    i=0
    for i in range(num_batch-1):# no (end, head) pair
        print('-------------------------------------------------------------  compare batch pair  (' +str(i)+','+str(i+1)+')')
        tii=time.time()

        
        print_len_of_batched_seeds_list(batched_seeds_list)    
        A_o=batched_seeds_list[i]
        B_o=batched_seeds_list[i+1]
        len_A_part,len_B_part = get_src_nodes_len(A_o,B_o)
        
        tij=time.time()
        print('\n\tpreparing two sides time : ' , time.time()-tii)
        InitializeBitList([A_o,B_o])
        print('\tInitializeBitList time : ' , time.time()-tij)
        tik=time.time()
        red_rate,ratio_A,ratio_B =getRedundancyRate(len_A_part,len_B_part,ideal_part_in_size) #r_cost=max(r_A, r_B)
        print('\tgetRedundancyCost: time  ' , time.time()-tik)
        print()
        print('\t\t\t\t\tlength of partitions '+ str(len_A_part)+', '+str(len_B_part))
        print()
        if red_rate < 1.0:
            continue
        # cutcost = getCost(A_o,B_o,bit_dict)
        # print('\tgetRedundancyCost: ' , time.time()-tik)
        # tih=time.time()
        # print('\t'+'-'*80)
        # max_neighbors=int(args.fan_out) # only for 1-layer model
        # local_nids=list(bit_dict.keys())
        # local_in_degrees = block_to_graph.in_degrees(local_nids).tolist()
        # max_in_degree=max(local_in_degrees) # only consider output nodes in degrees
        # maxDegree= max(max_neighbors, max_in_degree)
        # print('\t\t\t\tmaxDegree ',maxDegree)
        # bestRate=int_red_rate
        print('\tbefore terminate 1 the redundancy rate: ', red_rate)
        print('\t'+'-'*80)
        pass_=1
        for pass_step in range(pass_):
            if args.walkterm==1:
                ti=time.time()
                improvement,red_rate=walk_terminate_1(g, red_rate, args,ideal_part_in_size)
                # print('\twalk terminate 1 spend time', time.time()-ti)
                # print('\tafter terminate 1, whether it has improvement: ',improvement)
                # print('\tafter terminate 1 the  redundancy rate ',red_rate)
                if not improvement: # go several passes, until there is no improvement, 
                    
                    print('\tafter terminate 1, whether it has no improvement: ')
                    print('\tthe  redundancy rate is ',red_rate)
                    batched_seeds_list=update_Batched_Seeds_list(batched_seeds_list, bit_dict,i, i+1)
                    # update batched_seeds_list based on bit_dict
                    print('\tpass '+str(pass_step)+'  ')
                    src_len_list,_=get_partition_src_list_len(batched_seeds_list,ideal_part_in_size)
                    print_len_of_partition_list(src_len_list) 
                    # print out the result after exchange batched_seeds_list
                    
                    print()
                    break
            
        #--------------------- initialization checking done   ----------------   
        maxDegree=0
        gains={}
        bit_dict={}
        
        print('\t'+'-'*50 +'end of batch '+ str(i))
    # print(res)
   
    weight_list=get_weight_list(batched_seeds_list)
    len_list,redundancy_list=get_partition_src_list_len(batched_seeds_list,ideal_part_in_size)
    return batched_seeds_list, weight_list, len_list
 
def global_2_local(block_to_graph,batched_seeds_list):
    
    sub_in_nids = block_to_graph.srcdata['_ID'].tolist()
    # sub_out_nids = block_to_graph.dstdata['_ID'].tolist()
    global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
    # local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
    # ADJ_matrix=block_to_graph.adj_sparse('coo')
    t1=time.time()
    local_batched_seeds_list=[]
    for global_in_nids in batched_seeds_list:
        tt=time.time()
        local_in_nids = list(map(global_nid_2_local.get, global_in_nids))
        local_batched_seeds_list.append(local_in_nids)
    return local_batched_seeds_list

def local_2_global(block_to_graph,local_batched_seeds_list):
    
    sub_in_nids = block_to_graph.srcdata['_ID'].tolist()
    # sub_out_nids = block_to_graph.dstdata['_ID'].tolist()
    # global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
    local_nid_2_global = { i: sub_in_nids[i] for i in range(0, len(sub_in_nids))}
    # ADJ_matrix=block_to_graph.adj_sparse('coo')
    t1=time.time()
    global_batched_seeds_list=[]
    for local_in_nids in local_batched_seeds_list:
        tt=time.time()
        global_in_nids = list(map(local_nid_2_global.get, local_in_nids))
        global_batched_seeds_list.append(global_in_nids)
    return global_batched_seeds_list	
	

def  random_init_graph_partition(block_to_graph, args):
    tt = time.time()
    OUTPUT_NID, _ = torch.sort(block_to_graph.ndata[dgl.NID]['_N_dst'])
    batched_seeds_list,_ = gen_batched_seeds_list(OUTPUT_NID, args)
    
    
    t1 = time.time()
    batched_seeds_list=global_2_local(block_to_graph, batched_seeds_list) # global to local
    print('transfer time: ', time.time()-t1)
    #The graph_parition is run in block to graph local nids,it has no relationship with raw graph
    # batched_seeds_list,weights_list, p_len_list=graph_partition_quick(batched_seeds_list, block_to_graph, args)
    batched_seeds_list,weights_list, p_len_list=graph_partition_variant(batched_seeds_list, block_to_graph, args)
    batched_seeds_list=local_2_global(block_to_graph, batched_seeds_list) # local to global
    print('graph partition total spend ', time.time()-t1)
    t2=time.time()-tt
    return batched_seeds_list, weights_list,t2, p_len_list