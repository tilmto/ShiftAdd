from random import random, randint,shuffle
import numpy as np
import test_for_eyeriss as simnas
import time
from itertools import combinations,permutations
import copy
from  multiprocessing import Queue
import multiprocessing
import math

default_hw={ \
    'gb_vol':108*1024*8, \
    'rf_vol':6893, \
    'num_pe':168, \
    'num_rf':168
}

##############################
#shared util funcs
#############################

def compute_parallel(lp_order_string,tiling_string,hw_spec,stride):
    noc_consumption=1
    for i in tiling_string:
        if 'noc' in i:
            noc_consumption*=tiling_string[i]
    tmp_hw_spec=copy.deepcopy(hw_spec)
    ep=tmp_hw_spec['gb_vol']
    tmp_hw_spec['gb_vol']=int(tmp_hw_spec['gb_vol']/2)
    mp=tmp_hw_spec['gb_vol']
    sp=0
    p=life_eval(tiling_string,1,hw_spec,df_order=lp_order_string)
    while (ep-sp)>100 or (not p[1]):
        if p[1]:
            tmp_hw_spec['gb_vol']=math.floor((sp+mp)/2)
            ep=mp
            mp=tmp_hw_spec['gb_vol']
            p=life_eval(tiling_string,stride,tmp_hw_spec,df_order=lp_order_string)
        else:
            tmp_hw_spec['gb_vol']=math.ceil((mp+ep)/2)
            sp=mp
            mp=tmp_hw_spec['gb_vol']
            p=life_eval(tiling_string,stride,tmp_hw_spec,df_order=lp_order_string)

    parallem_num = np.floor(min(tmp_hw_spec['num_pe']/noc_consumption, hw_spec['gb_vol']/tmp_hw_spec['gb_vol']))

    return parallem_num


def life_eval(actions,stride,hw_spec,df_order=None):
    #function to query chip_estimator and get energy+latency feedback

    #actions: tiling factors for a specific loop-order
    #stride: the stride number for this CONV layer operation
    #hw_spec: hw specs for evaluation
    #df_order: loop-order for evaluation 
    #           !!!!if not provided PLS provide it in chip_estimator
    #           !!!!legacy functionality, so always try to provide specific loop-order here
    try:
        #input isolation
        input_actions=dict(actions)
        if df_order:
            input_df_order=list(df_order)
        else:
            input_df_order=None
        ene_results=simnas.sample_energy(input_actions,stride,hw_spec,input_df_order=input_df_order)
        penalty=-(ene_results[0]*1e-8*ene_results[1]*100)
        buffer_not_exceed=True
        #print(ene_results[0],ene_results[1])
    #if design hw constraint exceeded, 
    #if exceeded return extremely large penalty
    except Exception as e:
        if 'resource' in str(e):
            # print(e)
            pass
        else:
            print('error:',e)
            print(actions)
            print(df_order)         
        penalty=-9e12                                  #very strong penalty to over budget
        buffer_not_exceed=False
    return -penalty, buffer_not_exceed

actions= {'ch_out_rf':16, 'ch_in_rf':2, 'row_kernel_rf':5, 'ref_rf_we':64, 'row_out_rf':27, 'ref_rf_in':16, 'batch_rf':1,\
            'ref_rf_out':64, 'col_kernel_noc':5, 'ch_in_noc':1, 'col_out_noc':27, 'ch_out_noc':1,\
            'ref_gb_we':64, 'ch_out_gb':4, 'ref_gb_in':64, 'ch_in_gb':24,\
            'ref_gb_out':64, 'col_out_dram':1, 'ch_out_dram':4, 'batch_dram':4,\
            }    
stride=1
hw_spec=default_hw
df_order=['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf', 'ref_rf_we',\
                'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc',\
                'ref_gb_we', 'ch_out_gb', 'ref_gb_in',  'ch_in_gb', 'ref_gb_out', \
                'col_out_dram', 'ch_out_dram', 'batch_dram'\
               ]
actions={'ch_out_rf': 16, 'ch_in_rf': 3, 'row_kernel_rf': 3, 'ref_rf_out': 64, 'row_out_rf': 13, 'ref_rf_in': 16, 'batch_rf': 4, 'ref_rf_we': 64, 'col_kernel_noc': 3, 'ch_in_noc': 2, 'col_out_noc': 13, 'ch_out_noc': 2, 'ref_gb_we': 64, 'ch_out_gb': 2, 'ref_gb_in': 64, 'ch_in_gb': 32, 'ref_gb_out': 64, 'col_out_dram': 1, 'ch_out_dram': 4, 'batch_dram': 1} 
stride=1 
hw_spec={'gb_vol': 884736, 'rf_vol': 6893, 'num_pe': 168, 'num_rf': 168} 
df_order=['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf', 'ref_rf_we', 'col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc', 'ref_gb_we', 'ch_out_gb', 'ref_gb_in', 'ch_in_gb', 'ref_gb_out', 'col_out_dram', 'ch_out_dram', 'batch_dram']


life_eval(actions,stride,hw_spec,df_order=df_order)
    
def arch_life(child,input_stride_list,hw_spec,df_order=None):
    #evaluate the energy consumption for all layers in one network

    #child: a list of tiling factors dictionary --- [{},{},{},....],each corresponding to a layer
    #input_stride_list: a list of stride, each corresponding to a layer
    #hw_spec: hw spec used
    #df_order: loop-order in list of list, each inner list --- one single layer
    
    #return: total penalty for all layers and breakdown for each layer
    score=0
    #input isolation
    layer_wise=(type(df_order[0])==list)
    layer_break_down=[]
    raw_data=[]
    stride_list=list(input_stride_list)
    for i in range(len(child)):
        if not layer_wise:
            layer_score=life_eval(child[i],stride_list[i],hw_spec,df_order=df_order)                                                                  #######
        else:
            #print(child[i])
            #print(stride_list[i])
            layer_score=life_eval(child[i],stride_list[i],hw_spec,df_order=df_order[i])
        if layer_score > -9e12:
            layer_break_down.append(layer_score)
            score+=layer_score
            #raw_data.append(ene_results)   
        else:
            #print('layer:',i)
            layer_break_down.append(-9e12)
            #raw_data.append([9e12,9e12,{}])
            return -9e12,layer_break_down
    return score,layer_break_down
    
def pop_ranking(pop_list,score_board):
    #rank population list according to score_board, not working if directly using sorted(zip())
    #too ganky
    pop_indices=list(range(len(pop_list)))
    results = [(pop_indx,score_num) for score_num,pop_indx in sorted(zip(score_board,pop_indices),reverse=True)]
    pop_indices=[results[x][0] for x in list(range(len(results)))]
    tmp_pop=[]
    #maybe can get rid of the deepcopy
    for i in range(len(pop_list)):
        tmp_pop.append(pop_list[pop_indices[i]])
    pop_list=copy.deepcopy(tmp_pop)   
    score_board=[results[x][1] for x in list(range(len(results)))]
    return pop_list,score_board
    
    
#############################
#df order  specific
#############################
    
def lo_random_pop(layer_list=[10,7]):
    #randomly sample a specific loop-order
    #layer_list: denoting the number of components in each memory level, with NoC,RF levels excluded, as they are in a list pool to iterate 
    #            SRAM(GB) level
    #            DRAM level 
    #right now does not handle the repetitive population
    actions=[]
    for idx,size in enumerate(layer_list):
        for i in range(size-1,0,-1):
            actions.append(randint(0,i))
        actions.append(0)
    return actions
    
#print(random_pop([7,5,4,4]))
#exit()

def lo_give_birth(input_str1,input_str2):
    #combine two randomly generated loop-oders
    #with possiblity:
    #                 1. half features from str1 and half features from str2
    #                 2. entirely from str1
    #                 3. entirely from str2
    str1=list(input_str1)
    str2=list(input_str2)
    #random number to decide which part of the parents to inherit
    num=randint(0,2)
    if num==0:
        str3=str1[0:int(len(str1)/2)]+str2[int(len(str2)/2):]
    elif num==1:
        str3=str1
    else:
        str3=str2
    return str3

def lo_mutate(input_str1,prop,layer_list=[10,7]):
    #randomly permute some features of a input loop-oder
    #input_str1: input loop-order
    #prop: probability to mutate
    #layer_list: denoting number of components in each memory level
    str1=list(input_str1)
    if random()<=prop:
        #how many features under risk of mutation                              #currently 25 percent of features under risk of mutation
        size=int(0.25*(sum(layer_list)))
        if size<1:
            size=1
        pos=list(range(0,sum(layer_list)))
        #randomly pick 'size' of features
        shuffle(pos,random=random)
        pos=pos[0:size]
        ref_pos=[]
        #features number range for each feature 
        for i in layer_list:
            ref_pos+=list(range(i-1,0,-1))
            ref_pos+=[0]
        for i in pos:
            if ref_pos[i]==0:
                continue
            else:
                str1=str1[0:i]+[randint(0,ref_pos[i])]+str1[i+1:]
    return str1


#noc_template to be considered 
noc_template=[['col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc'], \
                      ['col_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      ['row_out_noc','col_out_noc','ch_out_noc'], \
                      ]

rf_noc_template=noc_template



def sample_results_df(input_actions,input_rf,layer_list=[10,7]):
    #translate index fomrat loop-order to string format 
    #input_actions: index format loop-order, output from lo_random_pop, lo_give_birth, lo_mutate
    #input_rf: rf_noc_template to take
    #layer_list: see above...
    actions=list(input_actions)
    df_dict ={
              #0:['col_kernel_noc', 'ch_in_noc', 'col_out_noc', 'ch_out_noc','row_kernel_noc','row_out_noc','batch_noc'], \
              0:['ref_gb_we','ch_out_gb', 'ref_gb_in','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb','ref_gb_out'], \
              1:['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
    }
    df_order=[]
    offset=0
    for idx,size in enumerate(layer_list):
        for i in range(offset,offset+size):
            try:
                ele=df_dict[idx][int(actions[i])]
            except:
                print('DATAFLOW INTEPRETATION ERROR!')
                print(actions)
                print(int(actions[i]))
                exit()            
            df_order.append(ele)
            df_dict[idx].remove(ele)  
        offset+=size
    return copy.deepcopy(input_rf)+df_order
#[[2, 2, 5, 4, 0, 3, 0, 2, 0, 0, 1, 4, 3, 1, 2, 0, 0], [5, 5, 3, 4, 0, 2, 3, 2, 0, 0, 0, 5, 2, 2, 1, 1, 0], [9, 7, 5, 5, 0, 2, 0, 0, 0, 0, 6, 3, 1, 0, 2, 0, 0], [3, 8, 0, 3, 0, 4, 3, 1, 0, 0, 6, 2, 2, 2, 0, 0, 0], [7, 6, 0, 2, 0, 3, 3, 2, 0, 0, 0, 1, 0, 0, 1, 1, 0]]
#print(sample_results_df([2, 2, 5, 4, 0, 3, 0, 2, 0, 0, 1, 4, 3, 1, 2, 0, 0], \
#['row_out_rf', 'col_kernel_rf', 'ch_out_rf', 'batch_rf', 'col_out_rf', 'ref_rf_we', 'ref_rf_in', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_kernel_noc', 'ch_in_noc', 'row_out_noc', 'ch_out_noc']))
#print(sample_results_df([6, 7, 6, 4, 1, 1, 0, 1, 0, 0, 6, 5, 4, 3, 2, 1, 0],['ch_out_rf', 'ch_in_rf', 'row_kernel_rf', 'ref_rf_out', 'row_out_rf', 'ref_rf_in', 'batch_rf', 'ref_rf_we', 'col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc']))

def arch_sample_results_df(dnn_layer_num,input_actions,input_rf,layer_list=[10,7]):
    #translate layers of input loop-order to string format
    input_actions=copy.deepcopy(input_actions)
    arch_df=[]
    for i in range(dnn_layer_num):
        arch_df.append(sample_results_df(input_actions[i],input_rf,layer_list=layer_list))
    return arch_df
        

#get the final results loop-order post_rf
#print(arch_sample_results_df(5,\
#                              [[8, 5, 3, 2, 5, 2, 3, 1, 0, 0, 5, 5, 1, 2, 0, 1, 0], [5, 3, 2, 4, 4, 4, 3, 1, 1, 0, 6, 3, 4, 1, 2, 0, 0], [3, 0, 0, 4, 1, 1, 1, 2, 0, 0, 5, 1, 0, 1, 0, 1, 0], [6, 7, 7, 5, 4, 4, 0, 0, 0, 0, 3, 1, 0, 3, 0, 0, 0], [8, 0, 0, 0, 0, 1, 0, 1, 1, 0, 5, 2, 2, 1, 0, 0, 0]],\
#                            ['row_out_rf', 'col_kernel_rf', 'ref_rf_out', 'row_kernel_rf', 'col_out_rf', 'batch_rf', 'ref_rf_we', 'ch_out_rf', 'ch_in_rf', 'ref_rf_in', 'row_kernel_noc', 'ch_in_noc', 'row_out_noc', 'ch_out_noc']))

def arch_lo_random_pop(dnn_layer_num,layer_list=[10,7]):
    #just leave it 
    arch_lo_pop=[]
    for _ in range(dnn_layer_num):
        arch_lo_pop.append(lo_random_pop(layer_list=layer_list))
    return arch_lo_pop
    

#######################
#layer level util func
#######################
#find the factors of a number
def r_factors(x):
    #find the factors of a number
    factor_list=[]
    for i in range(1, x + 1):
        if x % i == 0:
            factor_list.append(i)
    return factor_list
def diff_cal(factors):
    diff_sum=0
    for i in range(1,len(factors)):
        diff_sum+=abs(factors[i]-factors[i-1])
    return diff_sum
        
def factor_n(x,n=3,flexible_factor=1):
    #return the factor combo of length n for number x
    #flexible number:
    #               return factor combo of length n for number [x,flexible_factor)
    #               with requirement that the factors in in factor combo can not differ too much which is bad for resource partition


    #force one if n==1
    if n==1:
        flexible_factor=1
    #initialize max diff among factors and if this is original input or not
    diff_sum_min=math.inf
    input=True
    result=[]
    for _ in range(flexible_factor):
        #return factors of x
        factor_list=r_factors(x)
        num=factor_list[-1]
        tmp_list=[]
        for i in factor_list:
            for _ in range(n):
                tmp_list.append(i)
        # Get all combinations of factor_list
        # and length n
        comb = combinations(tmp_list, n) 
        for i in list(comb):
            mult=1
            for f in i:
                mult*=f
            if mult==num and (i not in result):               
                if input:
                    result.append(i)
                else:
                    if diff_cal(i)<diff_sum_min:
                        result.append(i)
                        diff_sum_min=diff_cal(i)
        if input:
            for i in result:
                tmp_diff_sum=diff_cal(i)
                if tmp_diff_sum<diff_sum_min:
                    diff_sum_min=tmp_diff_sum
        x+=1
        input=False
    return result

def permute_factor(input_factor_list):
    #permute the order within each factor in the factor_list
    #input  isolation
    factor_list=copy.deepcopy(input_factor_list)
    result=[]
    for f in factor_list:
        perm = permutations(f)     
        # Print the obtained permutations                        
        for i in list(perm): 
            if i not in result:                             
                result.append(i)
    return result


def random_pop_dict(config_dict,df_order,factor_list_dict):                      #ideally pop should not be a complete dict
    #randomly sample tiling factor combo for a specific loop-order

    #config_dict: with each key representing the data dimension: ch_in, ch_out,.... available
    #             with values representing the entire size of this dimension and how many levels we will tiling them into
    #df_order: loop-order 
    #factor_list_dict: possible tiling factor combos for each key in config_dict,
    #                  with value in list format representing factor combo of lengh = values in config_dict


    
    #########################################                                          #it adds up too much computation                                                     
    #the reason we did not fuse factor list into random pop
    #is that, we dont want to do all the permutation and ganky
    #stuff every time we call the random pop
    ##########################################
    
    df_dict={}
    #inefficient nested for loop....
    for key in config_dict.keys():
        #random sample a value combo
        try:
            pos=randint(0,len(factor_list_dict[key])-1)
        except:
            print('random pop dict erro exit')
            print(factor_list_dict)
            print(key)
            print(factor_list_dict[key])
            exit()
        value=factor_list_dict[key][pos]
        ctr=0
        for sub_key in df_order:
            if key in sub_key:
                df_dict[sub_key]=value[ctr]
                ctr+=1                    
            
    #return should be a complete df_dict
    return df_dict

    
def give_birth(str1,str2,config_dict):
    #combine or inherit the features of two tiling factos combo

    str3={}
    keys=list(config_dict.keys())
    #random number to decide which part of the parents to inherit
    num=randint(0,2)
    if num==0:
        #inherit from parent one
        for key in keys[0:int(len(keys)/2)]:
            for sub_key in str1.keys():
                if key in sub_key:
                    str3[sub_key]=str1[sub_key]                    
        #inherit from parent two
        for key in keys[int(len(keys)/2):]:
            for sub_key in str2.keys():
                if key in sub_key:
                    str3[sub_key]=str2[sub_key]
    elif num==1:
        str3=str1
    else:
        str3=str2
    
    #handling the special cases of kernel
    for key in str1.keys():
        if 'kernel' in key:
            str3[key]=11
            
    return str3


def mutate(str1,prop,config_dict,df_order,factor_list_dict):
    #with prop possiblity to randomly permute some features of a given tiling factor
    if random()<prop:
        keys=list(config_dict.keys())
        #currently only mutate one position
        pos=randint(0,len(keys)-1)
        key=keys[pos]
        sub_pos=randint(0,len(factor_list_dict[key])-1)
        ctr=0
        for sub_key in df_order:
            if key in sub_key:
                str1[sub_key]=factor_list_dict[key][sub_pos][ctr]
                ctr+=1
    return str1



#######################
#arch level util func
#######################
 

def arch_factor_list_dict(input_arch_config_dict):
    #generate all possible tiling factor combo for all layers 

    #input_arch_config_dict:
    #                       with each outer list representing each layer
    #                       with each dict within is a config_dict defined in random_pop_dict

    #input isolation
    arch_config_dict=copy.deepcopy(input_arch_config_dict)
    arch_factor_list=[]
    for config_dict in arch_config_dict:
        config_dict=config_dict[1]
        factor_list_dict={}
        for key in config_dict.keys():
            tmp_input=config_dict[key]
            #currently flexible_factor has a range of 5% of the original input
            #tmp_input.append(1)
            tmp_input.append(math.ceil((tmp_input[0]*0.05)))
            factor_list_dict[key]=permute_factor(factor_n(*tmp_input))
        arch_factor_list.append(factor_list_dict)
    return arch_factor_list

def arch_random_pop(input_arch_config_dict,input_df_order,input_arch_factor_list):
    #randomly sample tiling factors for layers of CONV
    #while random_pop_dict defined above is for single layer

    #input isolation
    arch_config_dict=copy.deepcopy(input_arch_config_dict)
    df_order=copy.deepcopy(input_df_order)
    arch_factor_list=copy.deepcopy(input_arch_factor_list)
    layer_wise=(type(df_order[0])==list)
    pop=[]
    layer=0
    for (factor_list,config_dict) in zip(arch_factor_list,arch_config_dict):
        #kernel_size=config_dict[2]
        config_dict=config_dict[1]
        if not layer_wise:
            pop.append(random_pop_dict(config_dict,df_order,factor_list))
        else:
            pop.append(random_pop_dict(config_dict,df_order[layer],factor_list))
        layer+=1
    return pop
def arch_give_birth(input_str1,input_str2,input_arch_config_dict):
    #input_isolation
    str1=copy.deepcopy(input_str1)
    str2=copy.deepcopy(input_str2)
    #arch wise inheritance
    str3=[]
    #random number to decide which part of the parents to inherit
    num=randint(0,2)
    if num==0:
        str3=str1[0:int(len(str1)/2)]+str2[int(len(str2)/2):]
    elif num==1:
        str3=str1
    else:
        str3=str2
    #layer wise inheritance                                                           #layer wise
    #......
            
    return str3
    
def arch_mutate(str1,prop, arch_factor_list,mutate_pos_num=4):
    #randomly permute input tiling factor combo features in all-layer scope
    #prop: probability of mutation
    #arch_factor_list: tiling factors choices pool
    #mutate_pos_num: number of position under potential mutation

    #max number of positions within one layer
    max_num=len(list(arch_factor_list[0].keys()))
    offset=0
    pos=[]
    for _ in range(len(arch_factor_list)):
        #randomly sample n positions 
        tmp_pos=list(range(offset,max_num+offset))
        shuffle(tmp_pos,random=random)
        pos+=list(tmp_pos[0:mutate_pos_num])
        offset+=max_num
    #pos=np.random.randint(max_num,size=mutate_pos_num)
    for indx in pos: 
        #decide which layer the pos belong
        layer=indx//len(list(arch_factor_list[0].keys()))
        #decide which part of the layer the pos belongs to
        in_layer=indx%len(list(arch_factor_list[0].keys()))         
        #fetch the part name ch_in, ch_out,col_out,row_out......
        key=list(arch_factor_list[layer].keys())[in_layer]
        #randomly sample a element from all the value combo in form of factors
        sub_pos=randint(0,len(arch_factor_list[layer][key])-1)
        element=arch_factor_list[layer][key][sub_pos]
        #assign the element back to str1
        ctr=0
        for i in str1[layer].keys():
            #i is the full name of the part: ch_in_rf,ch_out_noc,ch_in_noc.....
            if key in i:
                str1[layer][i]=element[ctr]
                ctr+=1   
    return str1

    
#####################    
#threading util
####################

def multi_p(func,args,output_q,num_worker_threads,dump_yard):
    #routine to distribute workers to multi cores
    #BETTER leave it

    #length of args has to be the multiple of num_worker_threads
    args=list(args)
    run_ites=int((len(args))//num_worker_threads)
    for run_ite in range(run_ites):
        processes = [multiprocessing.Process(target=func, args=([args[i]])) for i in range(run_ite*num_worker_threads,(run_ite+1)*num_worker_threads)]
        #print(len(processes))
        #print('queue size: ',score_pair.qsize())
        for p in processes:
            p.start()
        while not output_q.empty():
            pair=output_q.get()
            dump_yard.append(pair)
        for p in processes:
            p.join()
    while not output_q.empty():
        pair=output_q.get()
        dump_yard.append(pair)
    return None



###################################
#combined level
###################################
def gen_net_arch(df_order,dnn):
    #convert a dnn dimension input (from user) to a arch_config_dict format, according to given df_order
    #basically specfying how many levels of memory we want to tile the data dimnension to
    net_arch=copy.deepcopy(dnn)
    for i in df_order:
        if 'ch_out' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['ch_out'][1]+=1
        elif 'ch_in' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['ch_in'][1]+=1
        elif 'batch' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['batch'][1]+=1
        elif 'col_out' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['col_out'][1]+=1
        elif 'row_out' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['row_out'][1]+=1
        elif 'row_kernel' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['row_kernel'][1]+=1 
        elif 'col_kernel' in i:
            for layer in range(len(dnn)):
                net_arch[layer][1]['col_kernel'][1]+=1                
    return net_arch



#following func assume 3.7's order preservation in dict

#prevent overflow induced hanging in multi processing 
#so we need to streamline the communication between each process
def compress_dict(input_df_config_dict):
    df_config_dict=copy.deepcopy(input_df_config_dict)
    compressed=[]
    for layer in range(len(df_config_dict)):
        compressed.append([])
        for key in list(df_config_dict[layer].keys()):
            compressed[-1].append(df_config_dict[layer][key])
    return compressed
def decompress_dict(input_compressed,reference):
    compressed=copy.deepcopy(input_compressed)
    decompressed=copy.deepcopy(reference)
    for layer in range(len(compressed)):
        ctr=0
        for key in list(decompressed[layer].keys()):
            decompressed[layer][key]=compressed[layer][ctr]
            ctr+=1
    return decompressed
# dnn=[\
# (4, {'ch_out':(96,0),'ch_in':(3,0),'batch':(1,0),'col_out':(56,0),'row_out':(56,0)}, 11),\

# (1,{'ch_out':(256,0),'ch_in':(96,0),'batch':(1,0),'col_out':(27,0),'row_out':(27,0)}, 5),\

# (1,{'ch_out':(384,0),'ch_in':(384,0),'batch':(1,0),'col_out':(13,0),'row_out':(13,0)}, 3),\

# (1,{'ch_out':(256,0),'ch_in':(384,0),'batch':(1,0),'col_out':(13,0),'row_out':(13,0)}, 3),\

# (1,{'ch_out':(256,0),'ch_in':(384,0),'batch':(1,0),'col_out':(13,0),'row_out':(13,0)}, 3)\
# ]


def random_life(df_order, tiling_pool,input_stride_list,hw_spec,alloc_slots,rf_num,return_best_dict=False):
    #after smapling a loop-order, routine to optimize tiling factors to get the energy feedback     
    score_board=[]
    df_order=copy.deepcopy(df_order)
    #print(alloc_slots[rf_num],alloc_slots[rf_num+1])

    score_q=Queue()
    def worker(i):
        try: 
            score_q.put((arch_life(tiling_pool[i],input_stride_list,hw_spec,df_order=df_order)[0],i),False)
        except NameError:
            raise Exception("There is no room in the queue in rf template stage")
    if not score_q.empty():
        print('Some Trash in the score_q Queue')
        exit()

    work_load=list(range(alloc_slots[rf_num],alloc_slots[rf_num+1]))
    processes = [multiprocessing.Process(target=worker, args=([load])) for load in work_load]
    tmp_dump_yard=[]

    for p in processes:
        p.start()
        time.sleep(0.02)
    time.sleep(2)
    while not score_q.empty():
        tmp_batch=score_q.get()
        tmp_dump_yard.append(tmp_batch)
    for p in processes:
        p.join()
    #too many dump_yard...
    while not score_q.empty():
        tmp_batch=score_q.get()
        tmp_dump_yard.append(tmp_batch)

    score_pair=sorted(tmp_dump_yard,reverse=True)

    #for i in range(alloc_slots[rf_num],alloc_slots[rf_num+1]):
    #   
    #   #       tiling_for_all_layers=[]
    #   # for _ in range(len(df_order)):
    #   #     tiling_for_all_layers.append(i)
    #    score_board.append(arch_life(tiling_pool[i],input_stride_list,hw_spec,df_order=df_order)[0])
    #    print(len(score_board))
    #score_pair=sorted(zip(score_board,list(range(len(score_board)))),reverse=True)

    if return_best_dict:
        return score_pair[0][0], tiling_pool[score_pair[0][1]]
    else:
        return  score_pair[0][0]
def data_type_sizes(input_dnn):
    input_size=0
    output_size=0
    kernel_size=0
    for layer in input_dnn:
        input_size+=layer[1]['ch_in'][0]*layer[1]['row_out'][0]*layer[1]['col_out'][0]
        output_size+=layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*layer[1]['row_kernel'][0]
        kernel_size+=layer[1]['ch_out'][0]*layer[1]['row_out'][0]*layer[1]['col_out'][0]/layer[0]
    return input_size, output_size, kernel_size
    

    
class fpga_tiling_generator():
    def __init__(self, input_dnn, hw_spec):
        self.pe_array_dim_num=10
        input_dnn=copy.deepcopy(input_dnn)
        hw_spec=copy.deepcopy(hw_spec)
        [input_size,output_size, weight_size]=data_type_sizes(input_dnn)
        #self.total_rf=hw_spec['rf_vol']
        self.total_gb=hw_spec['gb_vol']
        self.dsp_limit=hw_spec['num_pe']
        #partition gb and rf proportionally towards each data type's size 
        #in the future if we would like to do dynamic allocation we can use the following as arguments instead
        #self.input_rf=input_size*self.total_rf//(input_size+output_size+weight_size)
        #self.output_rf=output_size*self.total_rf//(input_size+output_size+weight_size)
        #self.weight_rf=weight_size*self.total_rf//(input_size+output_size+weight_size)
        self.input_gb=input_size*self.total_gb//(input_size+output_size+weight_size)
        self.output_gb=output_size*self.total_gb//(input_size+output_size+weight_size)
        self.weight_gb=weight_size*self.total_gb//(input_size+output_size+weight_size)
        
        ch_in=[]
        ch_out=[]
        row_out=[]
        col_out=[]
        col_kernel=[]
        row_kernel=[]
        #collect all dimension information
        for layer in input_dnn:
            ch_in.append(layer[1]['ch_in'][0])
            ch_out.append(layer[1]['ch_out'][0])
            row_out.append(layer[1]['row_out'][0])
            col_out.append(layer[1]['col_out'][0])
            row_kernel.append(layer[1]['row_kernel'][0])
            col_kernel.append(layer[1]['col_kernel'][0])
        
        #weight stationary
        #noc standard:
        #row/col_kernel_noc=max(kernel_size)
        #increment channel until limit reached
        #rf gb standard. max out usage as possible
        self.ws_noc=[]
        self.ws_rf_gb_tiling_choices=[]
        self.ws_rf_gb_tiling_choices_num=[]
        for i in range(len(input_dnn)):
            self.ws_noc.append({})
            self.ws_rf_gb_tiling_choices.append({})
            self.ws_rf_gb_tiling_choices_num.append([])
            col_kernel_noc=input_dnn[i][1]['col_kernel'][0]
            row_kernel_noc=input_dnn[i][1]['row_kernel'][0]
            ch_in_choices=sorted(r_factors(input_dnn[i][1]['ch_in'][0]))
            ch_out_choices=sorted(r_factors(input_dnn[i][1]['ch_out'][0]))
            incr_out=True
            incr_out_index=0
            incr_in_index=0
            tmp_ch_in_noc=1
            tmp_ch_out_noc=1
            while col_kernel_noc*row_kernel_noc*tmp_ch_in_noc*tmp_ch_out_noc <= self.dsp_limit:
                ch_in_noc=tmp_ch_in_noc
                ch_out_noc=tmp_ch_out_noc
                if incr_out:
                    tmp_ch_out_noc=ch_out_choices[incr_out_index]
                    incr_out_index+=1
                    incr_out_index=min(len(ch_out_choices)-1, incr_out_index)
                    incr_out=(not incr_out)
                else:
                    tmp_ch_in_noc=ch_in_choices[incr_in_index]
                    incr_in_index+=1
                    incr_in_index=min(len(ch_in_choices)-1,incr_in_index)
                    incr_out=(not incr_out)
                if (incr_in_index == (len(ch_in_choices)-1)) and (incr_out_index == (len(ch_out_choices)-1)):
                    break
            self.ws_noc[-1]['ch_in_noc']=ch_in_noc
            self.ws_noc[-1]['ch_out_noc']=ch_out_noc
            self.ws_noc[-1]['row_kernel_noc']=row_kernel_noc
            self.ws_noc[-1]['col_kernel_noc']=col_kernel_noc
            self.ws_rf_gb_tiling_choices[-1]['ch_in_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_in'][0]//ch_in_noc,2)
            self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['ch_in_rf_gb_choices']))
            #self.ws_rf_gb_tiling_choices_num[-1].append(len(list(more_itertools.distinct_permutations(self.ws_rf_gb_tiling_choices[-1]['ch_in_rf_gb_choices']))))

            self.ws_rf_gb_tiling_choices[-1]['ch_out_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_out'][0]//ch_out_noc,2)
            self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['ch_out_rf_gb_choices']))
            #self.ws_rf_gb_tiling_choices_num[-1].append(len(list(more_itertools.distinct_permutations(self.ws_rf_gb_tiling_choices[-1]['ch_out_rf_gb_choices']))))
          
            self.ws_rf_gb_tiling_choices[-1]['col_out_rf_gb_choices']=factor_n(input_dnn[i][1]['col_out'][0],2)
            self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['col_out_rf_gb_choices']))
            #self.ws_rf_gb_tiling_choices_num[-1].append(len(list(more_itertools.distinct_permutations(self.ws_rf_gb_tiling_choices[-1]['col_out_rf_gb_choices']))))
 
            self.ws_rf_gb_tiling_choices[-1]['row_out_rf_gb_choices']=factor_n(input_dnn[i][1]['row_out'][0],2)
            self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['row_out_rf_gb_choices']))
            #self.ws_rf_gb_tiling_choices_num[-1].append(len(list(more_itertools.distinct_permutations(self.ws_rf_gb_tiling_choices[-1]['row_out_rf_gb_choices']))))

            self.ws_rf_gb_tiling_choices[-1]['batch_rf_gb_choices']=factor_n(input_dnn[i][1]['batch'][0],2)
            self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['batch_rf_gb_choices']))
            #self.ws_rf_gb_tiling_choices_num[-1].append(len(list(more_itertools.distinct_permutations(self.ws_rf_gb_tiling_choices[-1]['batch_rf_gb_choices']))))
        
            # self.ws_rf_gb_tiling_choices[-1]['ch_in_rf_gb_choices']=sorted(r_factors(input_dnn[i][1]['ch_in'][0]//ch_in_noc))
            # self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['ch_in_rf_gb_choices']))
            # self.ws_rf_gb_tiling_choices[-1]['ch_out_rf_gb_choices']=sorted(r_factors(input_dnn[i][1]['ch_out'][0]//ch_out_noc))
            # self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['ch_out_rf_gb_choices']))
            # self.ws_rf_gb_tiling_choices[-1]['col_out_rf_gb_choices']=sorted(r_factors(input_dnn[i][1]['col_out'][0]))
            # self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['col_out_rf_gb_choices']))
            # self.ws_rf_gb_tiling_choices[-1]['row_out_rf_gb_choices']=sorted(r_factors(input_dnn[i][1]['row_out'][0]))
            # self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['row_out_rf_gb_choices']))
            # self.ws_rf_gb_tiling_choices[-1]['batch_rf_gb_choices']=sorted(r_factors(input_dnn[i][1]['batch'][0]))
            # self.ws_rf_gb_tiling_choices_num[-1].append(len(self.ws_rf_gb_tiling_choices[-1]['batch_rf_gb_choices']))
            #print(self.ws_rf_gb_tiling_choices[-1]['ch_in_rf_gb_choices'])
            #print(self.ws_rf_gb_tiling_choices_num[-1])
            
            
            
            
            
        #output stationary
        self.os_noc=[]
        self.os_rf_gb_tiling_choices=[]
        self.os_rf_gb_tiling_choices_num=[]
        for i in range(len(input_dnn)):
            self.os_noc.append([])
            self.os_rf_gb_tiling_choices.append([])
            self.os_rf_gb_tiling_choices_num.append([])
            ch_out_choices=sorted(r_factors(input_dnn[i][1]['ch_out'][0]))
            col_out_choices=sorted(r_factors(input_dnn[i][1]['col_out'][0]))
            row_out_choices=sorted(r_factors(input_dnn[i][1]['row_out'][0]))
            
            
            pe_designs=[]
            str1=[]
            for l in ch_out_choices:
                for j in list(combinations(col_out_choices,2)):
                        if l*j[0]*j[1] <= self.dsp_limit:
                            pe_designs.append(j[0]*j[1]*l)
                            str1.append((l,j[0],j[1]))
                for k in col_out_choices:
                        if l*k*k <= self.dsp_limit:
                            pe_designs.append(k*k*l)
                            str1.append((l,k,k))
            pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            #print(pe_designs[1:10])
            
            for pe_design in pe_designs[0:self.pe_array_dim_num]:
                self.os_noc[-1].append({})
                self.os_rf_gb_tiling_choices[-1].append({})
                self.os_rf_gb_tiling_choices_num[-1].append([])
                self.os_noc[-1][-1]['ch_out_noc']=pe_design[0]
                self.os_noc[-1][-1]['col_out_noc']=pe_design[1]
                self.os_noc[-1][-1]['row_out_noc']=pe_design[2]
                
                self.os_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_in'][0],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']))
                
                self.os_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_out'][0]//self.os_noc[-1][-1]['ch_out_noc'],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']))
                
                self.os_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']=factor_n(input_dnn[i][1]['col_out'][0]//self.os_noc[-1][-1]['col_out_noc'],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']))
                
                self.os_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']=factor_n(input_dnn[i][1]['row_out'][0]//self.os_noc[-1][-1]['row_out_noc'],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']))
            
                self.os_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['col_kernel'][0],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']))    

                self.os_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['row_kernel'][0],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']))                 

                self.os_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']=factor_n(input_dnn[i][1]['batch'][0],2)
                self.os_rf_gb_tiling_choices_num[-1][-1].append(len(self.os_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']))


            
            

            
            
        #row stationary1 
        self.rs1_noc=[]
        self.rs1_rf_gb_tiling_choices=[]
        self.rs1_rf_gb_tiling_choices_num=[]
        for i in range(len(input_dnn)):
            self.rs1_noc.append([])
            self.rs1_rf_gb_tiling_choices.append([])
            self.rs1_rf_gb_tiling_choices_num.append([])
            ch_in_choices=sorted(r_factors(input_dnn[i][1]['ch_in'][0]))
            ch_out_choices=sorted(r_factors(input_dnn[i][1]['ch_out'][0]))
            col_out_choices=sorted(r_factors(input_dnn[i][1]['col_out'][0]))
            row_kernel_choices=sorted(r_factors(input_dnn[i][1]['row_kernel'][0]))
            
            
            pe_designs=[]
            str1=[]
            for m in ch_in_choices:
                for l in ch_out_choices:
                    for j in col_out_choices:
                        if m*l*j*input_dnn[i][1]['row_kernel'][0] <= self.dsp_limit:
                            pe_designs.append(m*l*j*input_dnn[i][1]['row_kernel'][0])
                            str1.append((m,l,j,input_dnn[i][1]['row_kernel'][0]))
            pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            #print(pe_designs[1:10])
            
            for pe_design in pe_designs[0:self.pe_array_dim_num]:
                self.rs1_noc[-1].append({})
                self.rs1_rf_gb_tiling_choices[-1].append({})
                self.rs1_rf_gb_tiling_choices_num[-1].append([])
                self.rs1_noc[-1][-1]['ch_in_noc']=pe_design[0]
                self.rs1_noc[-1][-1]['ch_out_noc']=pe_design[1]
                self.rs1_noc[-1][-1]['col_out_noc']=pe_design[2]
                self.rs1_noc[-1][-1]['row_kernel_noc']=pe_design[3]
                
                self.rs1_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_in'][0]//self.rs1_noc[-1][-1]['ch_in_noc'],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']))
                
                self.rs1_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_out'][0]//self.rs1_noc[-1][-1]['ch_out_noc'],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']))
                
                self.rs1_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']=factor_n(input_dnn[i][1]['col_out'][0]//self.rs1_noc[-1][-1]['col_out_noc'],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']))
                
                self.rs1_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']=factor_n(input_dnn[i][1]['row_out'][0],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']))
            
                self.rs1_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['col_kernel'][0],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']))    

                self.rs1_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']=[[1,1]]
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']))                 

                self.rs1_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']=factor_n(input_dnn[i][1]['batch'][0],2)
                self.rs1_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs1_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']))

            #noc_template to be considered 
# noc_template=[['col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc'], \
                      # ['col_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      # ['row_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      # ['row_out_noc','col_out_noc','ch_out_noc'], \
                      # ]
        
        
        #row stationary2
        self.rs2_noc=[]
        self.rs2_rf_gb_tiling_choices=[]
        self.rs2_rf_gb_tiling_choices_num=[]
        for i in range(len(input_dnn)):
            self.rs2_noc.append([])
            self.rs2_rf_gb_tiling_choices.append([])
            self.rs2_rf_gb_tiling_choices_num.append([])
            ch_in_choices=sorted(r_factors(input_dnn[i][1]['ch_in'][0]))
            ch_out_choices=sorted(r_factors(input_dnn[i][1]['ch_out'][0]))
            col_out_choices=sorted(r_factors(input_dnn[i][1]['col_out'][0]))
            col_kernel_choices=sorted(r_factors(input_dnn[i][1]['col_kernel'][0]))
            
            
            pe_designs=[]
            str1=[]
            for m in ch_in_choices:
                for l in ch_out_choices:
                    for j in col_out_choices:
                        if m*l*j*input_dnn[i][1]['col_kernel'][0] <= self.dsp_limit:
                            pe_designs.append(m*l*j*input_dnn[i][1]['col_kernel'][0])
                            str1.append((m,l,j,input_dnn[i][1]['col_kernel'][0]))
            pe_designs=[i[1] for i in sorted(zip(pe_designs,str1),reverse=True)]
            #print(pe_designs[1:10])
            
            for pe_design in pe_designs[0:self.pe_array_dim_num]:
                self.rs2_noc[-1].append({})
                self.rs2_rf_gb_tiling_choices[-1].append({})
                self.rs2_rf_gb_tiling_choices_num[-1].append([])
                self.rs2_noc[-1][-1]['ch_in_noc']=pe_design[0]
                self.rs2_noc[-1][-1]['ch_out_noc']=pe_design[1]
                self.rs2_noc[-1][-1]['col_out_noc']=pe_design[2]
                self.rs2_noc[-1][-1]['col_kernel_noc']=pe_design[3]
                
                self.rs2_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_in'][0]//self.rs2_noc[-1][-1]['ch_in_noc'],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['ch_in_rf_gb_choices']))
                
                self.rs2_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']=factor_n(input_dnn[i][1]['ch_out'][0]//self.rs2_noc[-1][-1]['ch_out_noc'],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['ch_out_rf_gb_choices']))
                
                self.rs2_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']=factor_n(input_dnn[i][1]['col_out'][0]//self.rs2_noc[-1][-1]['col_out_noc'],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['col_out_rf_gb_choices']))
                
                self.rs2_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']=factor_n(input_dnn[i][1]['row_out'][0],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['row_out_rf_gb_choices']))
            
                self.rs2_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']=[[1,1]]
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['col_kernel_rf_gb_choices']))    

                self.rs2_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']=factor_n(input_dnn[i][1]['row_kernel'][0],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['row_kernel_rf_gb_choices']))                 

                self.rs2_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']=factor_n(input_dnn[i][1]['batch'][0],2)
                self.rs2_rf_gb_tiling_choices_num[-1][-1].append(len(self.rs2_rf_gb_tiling_choices[-1][-1]['batch_rf_gb_choices']))

        
    def choices_generation(self,pe_array):
        if pe_array==0:
            return self.ws_rf_gb_tiling_choices_num,self.ws_rf_gb_tiling_choices
    def tiling_translation(self,layer,pe_array,pe_array_dim_choices,rf_gb_choices,rf_gb_choices_order):
        #pe_array 0-3
        #pe_array_dim_choices 0-self.pe_array_dim_num(10)
        #rf_gb_choices n*  5 for ws, 7 for os
        #rf_gb_choices_order 0-1 * 5 for ws, 7 for os
        #single layer
        #weight stationary
        if pe_array==0:
            ch_in_rf_gb_choice=self.ws_rf_gb_tiling_choices[layer]['ch_in_rf_gb_choices'][rf_gb_choices[0]]
            ch_out_rf_gb_choice=self.ws_rf_gb_tiling_choices[layer]['ch_out_rf_gb_choices'][rf_gb_choices[1]]
            col_out_rf_gb_choice=self.ws_rf_gb_tiling_choices[layer]['col_out_rf_gb_choices'][rf_gb_choices[2]]
            row_out_rf_gb_choice=self.ws_rf_gb_tiling_choices[layer]['row_out_rf_gb_choices'][rf_gb_choices[3]]
            batch_rf_gb_choice=self.ws_rf_gb_tiling_choices[layer]['batch_rf_gb_choices'][rf_gb_choices[4]]
            ch_in_tiling=list(permutations(ch_in_rf_gb_choice))[rf_gb_choices_order[0]]
            ch_out_tiling=list(permutations(ch_out_rf_gb_choice))[rf_gb_choices_order[1]]
            col_out_tiling=list(permutations(col_out_rf_gb_choice))[rf_gb_choices_order[2]]
            row_out_tiling=list(permutations(row_out_rf_gb_choice))[rf_gb_choices_order[3]]
            batch_tiling=list(permutations(batch_rf_gb_choice))[rf_gb_choices_order[4]]
            tiling_str=dict(self.ws_noc[layer])
            tiling_str['ch_in_gb']=ch_in_tiling[0]
            tiling_str['ch_out_gb']=ch_out_tiling[0]
            tiling_str['col_out_gb']=col_out_tiling[0]
            tiling_str['row_out_gb']=row_out_tiling[0]
            tiling_str['batch_gb']=batch_tiling[0]
            tiling_str['ch_in_dram']=ch_in_tiling[1]
            tiling_str['ch_out_dram']=ch_out_tiling[1]
            tiling_str['col_out_dram']=col_out_tiling[1]
            tiling_str['row_out_dram']=row_out_tiling[1]
            tiling_str['batch_dram']=batch_tiling[1]
            tiling_str['col_kernel_rf']=tiling_str['col_kernel_gb']=tiling_str['col_kernel_dram']=1
            tiling_str['row_kernel_rf']=tiling_str['row_kernel_gb']=tiling_str['row_kernel_dram']=1
        else:
            if pe_array==1:
                rf_gb_tiling_choices=self.rs2_rf_gb_tiling_choices
                noc=self.rs2_noc
            elif pe_array==2:
                rf_gb_tiling_choices=self.rs1_rf_gb_tiling_choices
                noc=self.rs1_noc
            elif pe_array==3:
                rf_gb_tiling_choices=self.os_rf_gb_tiling_choices
                noc=self.os_noc
            ch_in_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['ch_in_rf_gb_choices'][rf_gb_choices[0]]
            ch_out_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['ch_out_rf_gb_choices'][rf_gb_choices[1]]
            col_out_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['col_out_rf_gb_choices'][rf_gb_choices[2]]
            row_out_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['row_out_rf_gb_choices'][rf_gb_choices[3]]
            col_kernel_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['col_kernel_rf_gb_choices'][rf_gb_choices[4]]
            row_kernel_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['row_kernel_rf_gb_choices'][rf_gb_choices[5]]
            batch_rf_gb_choice=rf_gb_tiling_choices[layer][pe_array_dim_choices]['batch_rf_gb_choices'][rf_gb_choices[6]]
            ch_in_tiling=list(permutations(ch_in_rf_gb_choice))[rf_gb_choices_order[0]]
            ch_out_tiling=list(permutations(ch_out_rf_gb_choice))[rf_gb_choices_order[1]]
            col_out_tiling=list(permutations(col_out_rf_gb_choice))[rf_gb_choices_order[2]]
            row_out_tiling=list(permutations(row_out_rf_gb_choice))[rf_gb_choices_order[3]]
            col_kernel_tiling=list(permutations(col_kernel_rf_gb_choice))[rf_gb_choices_order[4]]
            row_kernel_tiling=list(permutations(row_kernel_rf_gb_choice))[rf_gb_choices_order[5]]
            batch_tiling=list(permutations(batch_rf_gb_choice))[rf_gb_choices_order[6]]
            tiling_str=dict(noc[layer][pe_array_dim_choices])
            tiling_str['ch_in_gb']=ch_in_tiling[0]
            tiling_str['ch_out_gb']=ch_out_tiling[0]
            tiling_str['col_kernel_gb']=col_kernel_tiling[0]
            tiling_str['row_kernel_gb']=row_kernel_tiling[0]
            tiling_str['col_out_gb']=col_out_tiling[0]
            tiling_str['row_out_gb']=row_out_tiling[0]
            tiling_str['batch_gb']=batch_tiling[0]
            tiling_str['ch_in_dram']=ch_in_tiling[1]
            tiling_str['ch_out_dram']=ch_out_tiling[1]
            tiling_str['col_out_dram']=col_out_tiling[1]
            tiling_str['row_out_dram']=row_out_tiling[1]
            tiling_str['col_kernel_dram']=col_kernel_tiling[1]
            tiling_str['row_kernel_dram']=row_kernel_tiling[1]
            tiling_str['batch_dram']=batch_tiling[1]
        return tiling_str
    def tiling_space_partition(self, pe_array,layer,pe_array_dim_choices):
        if pe_array==0:
            return self.ws_rf_gb_tiling_choices_num[layer]
        elif pe_array==1:
            return self.rs2_rf_gb_tiling_choices_num[layer][pe_array_dim_choices]
        elif pe_array==2:
            return self.rs1_rf_gb_tiling_choices_num[layer][pe_array_dim_choices]
        elif pe_array==3:
            return self.os_rf_gb_tiling_choices_num[layer][pe_array_dim_choices]
    

# def fpga_tiling_generator(input_dnn,buffer_limit,dsp_limit,return_partitioned_space=False,bit_width=16):
    # #current assumptions: at least a whole channel of ifmap, ofmap or kernel will be put into the buffer 
    # #low buffer utilization
    # #did not specify cross instance read/write...
    # tmp_layer=1
    # ch_in=[]
    # ch_out=[]
    # row_out=[]
    # col_out=[]
    # col_kernel=[]
    # row_kernel=[]
    # kernel_3_index=[]
    # layer_ctr=0
    # for layer in input_dnn:
        # ch_in.append(layer[1]['ch_in'][0])
        # ch_out.append(layer[1]['ch_out'][0])
        # row_out.append(layer[1]['row_out'][0])
        # col_out.append(layer[1]['col_out'][0])
        # row_kernel.append(layer[1]['row_kernel'][0])
        # col_kernel.append(layer[1]['col_kernel'][0])
        # if layer[1]['row_kernel'][0] ==3:
            # kernel_3_index.append(layer_ctr)
        # layer_ctr+=1

    # try:
        # if len(ch_in)>1:
            # ch_in.remove(3)
    # except:
        # pass
    # ch_out_bram=_gcd(ch_out)
    # ch_in_bram=_gcd(ch_in)

    # col_out_bram=_gcd(col_out)
    # row_out_bram=_gcd(row_out)
    # row_kernel_bram=max(row_kernel)
    # col_kernel_bram=max(col_kernel)

 # #buffer size calc
    # output_b_size=ch_out_bram*col_out_bram*row_out_bram*bit_width
    # input_b_size=ch_in_bram*row_out_bram*col_out_bram*bit_width
    # weight_b_size=ch_in_bram*ch_out_bram*col_kernel_bram*row_kernel_bram*bit_width
    # f_index=0
    # out_ch_decr=False
    # while (output_b_size+input_b_size+weight_b_size) > buffer_limit:
        # if (len(input_dnn)>1):
            # raise Exception('Buffer exceeded') 
        # print('buffer exceeded, retry tiling')
        # f_index+=1
        # if out_ch_decr:
            # ch_out_bram=sorted(r_factors(ch_out[0]),reverse=True)[min(len(r_factors(ch_out[0]))-1,f_index)]
            # out_ch_decr=(not out_ch_decr)
        # else:
            # ch_in_bram=sorted(r_factors(ch_in[0]),reverse=True)[min(len(r_factors(ch_in[0]))-1,f_index)]
            # out_ch_decr=(not out_ch_decr)
        # output_b_size=ch_out_bram*col_out_bram*row_out_bram*bit_width
        # input_b_size=ch_in_bram*row_out_bram*col_out_bram*bit_width
        # weight_b_size=ch_in_bram*ch_out_bram*col_kernel_bram*row_kernel_bram*bit_width

    # print((output_b_size+input_b_size+weight_b_size)/8/1024, 'kB buffer used')
# #    if (output_b_size+input_b_size+weight_b_size) > buffer_limit:
# #        raise Exception('buffer size exceeded')
    # #    
    # dram_tiling_head=[]
    # for layer in input_dnn:
        # dram_tiling_head.append({})
        # dram_tiling_head[-1]['ch_out_dram']=max(layer[1]['ch_out'][0]/ch_out_bram,1)
        # dram_tiling_head[-1]['ch_in_dram']=max(layer[1]['ch_in'][0]/ch_in_bram,1)
        # dram_tiling_head[-1]['row_out_dram']=max(layer[1]['row_out'][0]/row_out_bram,1)
        # dram_tiling_head[-1]['col_out_dram']=max(layer[1]['col_out'][0]/col_out_bram,1)
        # dram_tiling_head[-1]['col_kernel_dram']=max(layer[1]['col_kernel'][0]/col_kernel_bram,1)
        # dram_tiling_head[-1]['row_kernel_dram']=max(layer[1]['row_kernel'][0]/row_kernel_bram,1)
        # dram_tiling_head[-1]['batch_dram']=1

    # noc_template=[['col_kernel_noc','row_kernel_noc','ch_in_noc','ch_out_noc'], \
                      # ['col_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      # ['row_kernel_noc','ch_in_noc','col_out_noc','ch_out_noc'], \
                      # ['row_out_noc','col_out_noc','ch_out_noc'], \
                      # ]

    # bram_noc_tiling=[]

    # #1
    # col_kernel_noc=sorted(r_factors(col_kernel_bram))
    # row_kernel_noc=sorted(r_factors(row_kernel_bram))
    # ch_in_noc=sorted(r_factors(ch_in_bram))
    # ch_out_noc=sorted(r_factors(ch_out_bram))
    # for i in col_kernel_noc:
        # for j in row_kernel_noc:
            # for k in ch_in_noc:
                # for l in ch_out_noc:
                    # if i*j*k*l<=dsp_limit:
                        # bram_noc_tiling.append({})
                        # bram_noc_tiling[-1]['batch_gb']=1
                        # bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/l
                        # bram_noc_tiling[-1]['ch_out_noc']=l
                        # bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram/k
                        # bram_noc_tiling[-1]['ch_in_noc']=k
                        # bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram/j
                        # bram_noc_tiling[-1]['row_kernel_noc']=j
                        # bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram/i
                        # bram_noc_tiling[-1]['col_kernel_noc']=i
                        # bram_noc_tiling[-1]['row_out_gb']=row_out_bram
                        # bram_noc_tiling[-1]['col_out_gb']=col_out_bram
                    # elif return_partitioned_space:
                        # bram_noc_tiling.append({})
                        # bram_noc_tiling[-1]['batch_gb']=1
                        # bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/l
                        # bram_noc_tiling[-1]['ch_out_noc']=l
                        # bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram/k
                        # bram_noc_tiling[-1]['ch_in_noc']=k
                        # bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram/j
                        # bram_noc_tiling[-1]['row_kernel_noc']=j
                        # bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram/i
                        # bram_noc_tiling[-1]['col_kernel_noc']=i
                        # bram_noc_tiling[-1]['row_out_gb']=row_out_bram
                        # bram_noc_tiling[-1]['col_out_gb']=col_out_bram
    # alloc_slots=[0]
    # alloc_slots.append(len(bram_noc_tiling))
    # space_partition=[]
    # space_partition.append([len(col_kernel_noc),len(row_kernel_noc),len(ch_in_noc),len(ch_out_noc)])
    # #2
    # col_out_noc=sorted(r_factors(col_out_bram))
    # for i in col_kernel_noc:
        # for j in col_out_noc:
            # for k in ch_in_noc:
                # for l in ch_out_noc:
                    # if i*j*k*l<=dsp_limit:  
                        # bram_noc_tiling.append({})
                        # bram_noc_tiling[-1]['batch_gb']=1
                        # bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/l
                        # bram_noc_tiling[-1]['ch_out_noc']=l
                        # bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram/k
                        # bram_noc_tiling[-1]['ch_in_noc']=k
                        # bram_noc_tiling[-1]['col_out_gb']=col_out_bram/j
                        # bram_noc_tiling[-1]['col_out_noc']=j
                        # bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram/i
                        # bram_noc_tiling[-1]['col_kernel_noc']=i
                        # bram_noc_tiling[-1]['row_out_gb']=row_out_bram
                        # bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram
                    # elif return_partitioned_space:
                        # bram_noc_tiling.append({})
                        # bram_noc_tiling[-1]['batch_gb']=1
                        # bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/l
                        # bram_noc_tiling[-1]['ch_out_noc']=l
                        # bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram/k
                        # bram_noc_tiling[-1]['ch_in_noc']=k
                        # bram_noc_tiling[-1]['col_out_gb']=col_out_bram/j
                        # bram_noc_tiling[-1]['col_out_noc']=j
                        # bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram/i
                        # bram_noc_tiling[-1]['col_kernel_noc']=i
                        # bram_noc_tiling[-1]['row_out_gb']=row_out_bram
                        # bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram


    # alloc_slots.append(len(bram_noc_tiling))
    # space_partition.append([len(col_kernel_noc),len(col_out_noc),len(ch_in_noc),len(ch_out_noc)])
    # #3
    # for i in row_kernel_noc:
        # for j in col_out_noc:
            # for k in ch_in_noc:
                # for l in ch_out_noc:
                    # if i*j*k*l<=dsp_limit:
                        # bram_noc_tiling.append({})
                        # bram_noc_tiling[-1]['batch_gb']=1
                        # bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/l
                        # bram_noc_tiling[-1]['ch_out_noc']=l
                        # bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram/k
                        # bram_noc_tiling[-1]['ch_in_noc']=k
                        # bram_noc_tiling[-1]['col_out_gb']=col_out_bram/j
                        # bram_noc_tiling[-1]['col_out_noc']=j
                        # bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram/i
                        # bram_noc_tiling[-1]['row_kernel_noc']=i
                        # bram_noc_tiling[-1]['row_out_gb']=row_out_bram
                        # bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram
                    # elif return_partitioned_space:
                        # bram_noc_tiling.append({})
                        # bram_noc_tiling[-1]['batch_gb']=1
                        # bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/l
                        # bram_noc_tiling[-1]['ch_out_noc']=l
                        # bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram/k
                        # bram_noc_tiling[-1]['ch_in_noc']=k
                        # bram_noc_tiling[-1]['col_out_gb']=col_out_bram/j
                        # bram_noc_tiling[-1]['col_out_noc']=j
                        # bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram/i
                        # bram_noc_tiling[-1]['row_kernel_noc']=i
                        # bram_noc_tiling[-1]['row_out_gb']=row_out_bram
                        # bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram
    # alloc_slots.append(len(bram_noc_tiling))
    # space_partition.append([len(row_kernel_noc),len(col_out_noc),len(ch_in_noc),len(ch_out_noc)])
    # #4
    # row_out_noc=sorted(r_factors(row_out_bram))
    # for i in col_out_noc:
        # for j in row_out_noc:
            # for k in ch_out_noc:
                # if i*j*k <= dsp_limit:
                    # bram_noc_tiling.append({})
                    # bram_noc_tiling[-1]['batch_gb']=1
                    # bram_noc_tiling[-1]['col_out_gb']=col_out_bram/i
                    # bram_noc_tiling[-1]['col_out_noc']=i
                    # bram_noc_tiling[-1]['row_out_gb']=col_out_bram/j
                    # bram_noc_tiling[-1]['row_out_noc']=j
                    # bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/k
                    # bram_noc_tiling[-1]['ch_out_noc']=k
                    # bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram
                    # bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram
                    # bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram
                # elif return_partitioned_space:
                    # bram_noc_tiling.append({})
                    # bram_noc_tiling[-1]['batch_gb']=1
                    # bram_noc_tiling[-1]['col_out_gb']=col_out_bram/i
                    # bram_noc_tiling[-1]['col_out_noc']=i
                    # bram_noc_tiling[-1]['row_out_gb']=col_out_bram/j
                    # bram_noc_tiling[-1]['row_out_noc']=j
                    # bram_noc_tiling[-1]['ch_out_gb']=ch_out_bram/k
                    # bram_noc_tiling[-1]['ch_out_noc']=k
                    # bram_noc_tiling[-1]['row_kernel_gb']=row_kernel_bram
                    # bram_noc_tiling[-1]['col_kernel_gb']=col_kernel_bram
                    # bram_noc_tiling[-1]['ch_in_gb']=ch_in_bram
    # alloc_slots.append(len(bram_noc_tiling))
    # space_partition.append([len(col_out_noc),len(row_out_noc),len(ch_out_noc)])
    # result_tiling_pool=[]
    # for i in bram_noc_tiling:
        # result_tiling_pool.append(copy.deepcopy(dram_tiling_head))
        # for j in range(len(result_tiling_pool[-1])):
            # result_tiling_pool[-1][j]=dict(list(result_tiling_pool[-1][j].items())+list(i.items()))

    # for i in result_tiling_pool:
        # for j in kernel_3_index:
            # try:
                # if i[j]['row_kernel_gb']==5:
                    # i[j]['row_kernel_gb']=3
            # except:
                # pass
            # try:
                # if i[j]['row_kernel_noc']==5:
                    # i[j]['row_kernel_noc']=3
            # except:
                # pass
            # try:
                # if i[j]['col_kernel_gb']==5:
                    # i[j]['col_kernel_gb']=3
            # except:
                # pass
            # try:
                # if i[j]['col_kernel_noc']==5:
                    # i[j]['col_kernel_noc']=3
            # except:
                # pass
    # if len(result_tiling_pool[0])==1:
        # pass
    # else:
        # for i in result_tiling_pool:
            # try:
                # i[0]['ch_in_noc']=1
            # except:
                # pass
            # i[0]['ch_in_gb']=3
            # i[0]['ch_in_dram']=1
    # if not return_partitioned_space:
        # return result_tiling_pool,alloc_slots
    # else:
        # return result_tiling_pool,alloc_slots, space_partition

def _gcd(l):
    if len(l)==1:
        return l[0]
    def find_gcd(x, y): 
        while(y): 
            x, y = y, x % y 
      
        return x 

      
    num1=l[0] 
    num2=l[1] 
    gcd=find_gcd(num1,num2) 
      
    for i in range(2,len(l)): 
        gcd=find_gcd(gcd,l[i]) 
    return gcd

