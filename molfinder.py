#!/usr/bin/env python3

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import RDConfig
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import  AllChem
from feature_calculation import calculate_feature
from LNP_model import AI_LNP_model
import random
import re


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str,default='./XH_lipid.csv', help="")
parser.add_argument("-r",
                    "--random-seed",
                    metavar="N",
                    type=int,
                    default=None,
                    help="Random seed")
parser.add_argument(
    "--bank-size",
    metavar="N",
    type=int,
    default=50,
    help="",
)
parser.add_argument(
    "--seed-size",
    metavar="N",
    type=int,
    default=20,
    help="",
)
parser.add_argument(
    "--max-round",
    metavar="N",
    type=int,
    default=150,
    help="",
)
parser.add_argument(
    "-cvg",
    "--convergent-round",
    metavar="N",
    default=10,
    type=int,
    help=
    "Convergent round; It determines when D_cut reaches a minimum value. And also It decides diversity of molecules",
)
parser.add_argument(
    "-c",
    "--coefficient",
    metavar="Float",
    type=float,
    default=0.9,
    help="coefficient of reward function.",
)
parser.add_argument(
    "-dist",
    "--dist-coef",
    metavar="coef. of distance",
    type=float,
    default=0.90,
    help="Control Dcut",
)
parser.add_argument(
    "--target",
    metavar="SMILES",
    type=str,
    default=None,
    help="target_moleclue SMILES",
)
parser.add_argument(
    "-fp",
    "--fp-method",
    type=str,
    default="ECFP6_2048",
    help="Select Fingerprint Method (rdkit/morgan)",
)
# parser.add_argument(
#     "-nf", "--nfeatures", metavar="N", type=int, default=2, help="a number of features"
# )
parser.add_argument("-v",
                    "--verbosity",
                    action="count",
                    default=0,
                    help="print error")
args = parser.parse_args()

if args.verbosity == 0:
    rdBase.DisableLog('rdApp.*')

fp_method = args.fp_method

if fp_method == "rdkit":
    _get_fp = lambda x: Chem.RDKFingerprint(x)
elif fp_method == "morgan":
    _get_fp = lambda x: AllChem.GetMorganFingerprint(x, 2)
elif fp_method == "ECFP6_1024":
    _get_fp = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=1024)
elif fp_method == "ECFP6_2048":
    _get_fp = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3)    


def get_fp(mol_or_smi):
    if type(mol_or_smi) in [Chem.rdchem.Mol, Chem.rdchem.RWMol]:
        _mol = mol_or_smi
    elif type(mol_or_smi) == str:
        _mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        raise ValueError("This type is not allowed.")
    return _get_fp(_mol)


# -------------------------------------------------------- #
"""
@@FEATURES
Features must return a number.
1. Set nfeatures
2. Add your functions
"""




    
model_output = lambda f :AI_LNP_model(f)     #model input :[smiles,feature]



def cal_features(part_array):  # set data_column
    mol_feature = calculate_feature(part_array[:,:6])    #mol_feature return Datafream
    return model_output(mol_feature)

def obj_fn(b):                                        #objective function
    if b.ndim ==1:
        b_feature = calculate_feature(b[:6])
    else:
        b_feature = calculate_feature(b[:,:6])
    return model_output(b_feature)


# -------------------------------------------------------- #


class ChkTime:
    def __init__(self):
        self.t = time.time()

    def get(self):
        return (time.time() - self.t) / 60


def cal_avg_dist(solutions):          #averge similarity             

    dist_sum = 0
    min_dist = 10
    max_dist = 0
    _n = len(solutions)

    for i in range(_n - 1):
        for j in range(i + 1, _n):
            fps1 = get_fp(solutions[i, 6])
            fps2 = get_fp(solutions[j, 6])
            dist = TanimotoSimilarity(fps1, fps2)
            dist_sum += dist
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

    return dist_sum / (_n * (_n - 1) / 2)  


def init_bank(file_name, nbank=None, nsmiles=None, rseed=5):      #nsmile : num of seed molecule 

    np.random.seed(rseed)

    _df = pd.read_csv(file_name)
    df = _df[:nsmiles].values

    shuffled_index = np.random.permutation(len(df))
    tmp_bank = df[shuffled_index][:nbank]
    df = pd.DataFrame(tmp_bank, columns=_df.columns.tolist())
    df.to_csv('./init_bank.csv',index=False)


    bank = np.empty([tmp_bank.shape[0], 8], dtype=object)
    bank[:, :6] = tmp_bank[:, :6]  # SMILES、head、tail1、tail2、tail3、tail4
    bank[:, 7] = True  # usable label True
    smi = bank[:,0]
    bank[:,6] =[Chem.MolFromSmiles(s) for s in smi]


    return bank


def prepare_seed(solutions, seed):      

    solutions = solutions[np.where(solutions[:, 7] == True)]  
    x = np.argsort(solutions[:,7])     #kw: why sorting on this column (all True) ??
    solutions = x[::-1]                   



    if len(solutions) >= nseed:
        i = 0
        if len(seed) is 0:  # First selection,
            bank[solutions[0], 7] = False # kw: is bank equal to solutions?
            seed.append(bank[solutions[0]])
            i += 1

        if len(solutions) < len(seed):
            print(
                f"## Solutions is less than seeds / round {round_} / iter {niter}"
            )
            raise ValueError

        while len(seed) < nseed:
            if len(solutions) == i + 1:
                print(
                    f"## Solutions is empty state / unused > nseed / # of seed: {len(seed)} / round {round_} / iter {niter}"
                )
                break
                # raise ValueError
            bank[solutions[i], 7] = False
            seed.append(bank[solutions[i]])
            i += 1


    print(f"@ prepare_seed finished!")
    return np.asarray(seed)



def extract_and_remove_brackets(input_string):
    parenthese = []
    stack = []
    output_string = ""
    for char in input_string:
        if char == '(':
            stack.append('')
        elif char == ')':
            if stack:
                parenthese.append(''.join(stack.pop()))
        else:
            if stack:
                stack[-1] += char
            else:
                output_string += char
    return  parenthese, output_string


def reverse_tail2 (s): #kw: need more comments!
    d = {}
    p_part,main_part = extract_and_remove_brackets(s)
    reverse_main = main_part[::-1]
    for i,c in enumerate(s):
        if c == '(':
            d[s[i+1]] = i
    if len(d) == 2:
        k = list(d.keys())[0]
        v = d[k]
        if k == '=':
            if v == 1:
                new_main = reverse_main + '(=O)'
            else:
                new_main = reverse_main[:-v + 1] + '(=O)' + reverse_main[-v + 1:]
            next_value = d.get(list(d.keys())[-1])
            new_tail2 = new_main[:-next_value + 1] +'(' +p_part[1]+')' + new_main[-next_value + 1:]
        else:
            new_main = reverse_main[:-v+1] + '('+p_part[0] +')'+reverse_main[-v+1:]
            next_value = d['=']
            new_tail2 = new_main[:-next_value+1] + '(=O)' + new_main[-next_value+1:]
        return  new_tail2
    elif len(d) ==1:
        k = list(d.keys())[0]
        v = d[k]
        if k == '=':
            if v == 1:
                new_tail2 = reverse_main + '(=O)'
            else:
                new_tail2 = reverse_main[:-v + 1] + '(=O)' + reverse_main[-v + 1:]
        else:
            new_tail2 = reverse_main[:-v+1 ] + '(' + p_part[0] + ')' + reverse_main[-v +1 :]
        return new_tail2
    else:
        return  reverse_main
def insert_tail1(head, tail1):
    return head[:1] + '('+ tail1 + ')' + head[1:]

def insert_tail2(tail1_lipid, tail2):
    tail2 = reverse_tail2(tail2)
    return tail2+ tail1_lipid

def combination_lipid(head,tail1, tail2):   #combine head ,tail1,tail2
    tail1_lipid = insert_tail1(head,tail1)
    new_lipid = insert_tail2(tail1_lipid,tail2)
       
    return [new_lipid,head,tail1,tail2,None,None] #tail3 tail4 为0



def append_seed(new_part, update_solution):
    new_mol = Chem.MolFromSmiles(new_part[0])
    new_tail1_mol =Chem.MolFromSmiles(new_part[1])
    new_tail2_mol =Chem.MolFromSmiles(new_part[2])
    if new_mol and new_tail1_mol and new_tail2_mol:
        update_solution.append(new_part)
        return 1
    else:
        return 0
def replace_funtiongroup(s, target,target1):       #replace function-group in tail # kw: need more comments
    stack = []
    replace_list = []
    result = list(s)

    for i, char in enumerate(s):       
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
        elif not stack:
            j = i
            if j  < len(s)-1 :         
                if s[j + 1] != '('  :          
                    replace_list.append(j)
    mean = np.mean(replace_list)
    std_dev = np.std(replace_list)
    weights = [1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2) for x in replace_list[1:]]
    replace_set = set()
    while len(replace_set) <2:
        replace_set.add(random.choices(replace_list[1:], weights)[0])
    replace_list = list(replace_set)
    result[replace_list[0]] = target
    if target1== '/C=C\\' and replace_list[1] ==0:
        result[replace_list[1]] = 'C=C'
    else:
        result[replace_list[1]] = target1
    return ''.join(result)

def ester_replace(smiles):   
    mol = Chem.MolFromSmiles(smiles)
    if mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)O')) or  mol.HasSubstructMatch(Chem.MolFromSmarts('[N][C](=[O])')):
            if 'OC(=O)' in smiles:
                smiles= smiles.replace('OC(=O)', '')
            elif 'C(=O)O' in smiles:
                smiles= smiles.replace('C(=O)O', '')
            elif  'NC(=O)' in  smiles:
                smiles= smiles.replace('NC(=O)', '')
            elif 'C(=O)N'in  smiles:
                smiles= smiles.replace('C(=O)N', '')
    if mol.HasSubstructMatch(Chem.MolFromSmiles('C=C')):
            smiles = smiles.replace('C=C', 'CC')
            return smiles.replace("/", "").replace("\\", "")
    #kw: how about SS?
    else:
        return smiles
def group_direction(group):
    seed_num = random.random()             #0-1 random
    if group == 'C(=O)O' :
        if seed_num < 0.5:
            return 'C(=O)O'
        else:
            return 'OC(=O)'
    elif group == 'C(=O)N':
        if seed_num < 0.5:
            return 'C(=O)N'
        else:
            return 'NC(=O)'
    else:
        return 'C'

def prepare_child(seed,
                  head,
                  nCross1=2,
                  nCross2=2,
                  nReplace=2,
                  nAdd=2,
                  nRemove=2):
    update_solution = []
  
    for i in range(seed.shape[0]): 
        j = 0
        q = 0 
        for h in head:
            while j <nCross1:    #tail1 cross-over
                        tail1 = seed[i,2]
                        w = np.random.randint(len(bank))
                        tail2 = bank[w, 3]
                        if np.random.random() >= 0.5:
                            new_lipid_part = combination_lipid(h, tail1,tail2)
                        else:
                            new_lipid_part = combination_lipid(h, tail2,tail1)

                        j += append_seed(new_lipid_part, update_solution)
                        bank[w, 7] = False
            while q <nCross2:   #tail2 cross-over
                        tail2 = seed[i,3]
                        w = np.random.randint(len(bank))
                        tail1 = bank[w, 2]
                        if np.random.random() >= 0.5:
                            new_lipid_part = combination_lipid(h, tail1,tail2)
                        else:
                            new_lipid_part = combination_lipid(h, tail2,tail1)

                        q += append_seed(new_lipid_part, update_solution)
                        bank[w, 7] = False

            # REPLACE ###                             
        q = 0
        j = 0
        while j < nReplace:     #tail1 replacement
                h = seed[i,1]
                tail1 = seed[i,2]
                tail2 = seed[i,3]
                group = ['C(=O)O','C(=O)N','CC']
                group1 = ['/C=C\\','SS','CC']
                random_group = random.choices(group,weights = [0.85,0.05,0.1],k=1)
                group_list = group_direction(random_group[0])
                group1_list = random.choices(group1,weights=[0.125,0.125,0.75],k=1)
                new_tail1 = replace_funtiongroup(ester_replace(tail1),group_list,group1_list[0])
                new_lipid_part = combination_lipid(h, new_tail1,tail2)
                j += append_seed(new_lipid_part, update_solution)
        while q < nReplace:   #tail2 replacement
                h = seed[i,1]
                tail1 =seed[i,2]
                tail2 = seed[i,3]
                group = ['C(=O)O','C(=O)N','CC'] # kw: if select CC, adding 2 carbons?
                group1 = ['/C=C\\','SS','CC']   # kw: if select CC, adding 2 carbons?
                random_group = random.choices(group,weights = [0.85,0.05,0.1],k=1)
                group_list = group_direction(random_group[0])
                group1_list = random.choices(group1,weights=[0.125,0.125,0.75],k=1)
                new_tail2 = replace_funtiongroup(ester_replace(tail2),group_list,group1_list[0])
                new_lipid_part = combination_lipid(h, tail1,new_tail2)
                q += append_seed(new_lipid_part, update_solution)
            # ADD ###
        q = 0
        j = 0
        while j <nAdd:       #tail1 add C atom
                h = seed[i,1]
                tail1 =seed[i,2]
                tail2 = seed[i,3]
                ope_index = [m.start() for m in re.finditer('CC', tail1)]
                random_index = random.choice(ope_index)
                new_tail1 =  tail1[:random_index] + 'C' +tail1[random_index:]
                
                new_lipid_part = combination_lipid(h, new_tail1,tail2)
                j += append_seed(new_lipid_part, update_solution)
        while q <nAdd:   #tail1 add C atom
                h = seed[i,1]
                tail1 =seed[i,2]
                tail2 = seed[i,3]
                ope_index = [m.start() for m in re.finditer('CC', tail2)]
                random_index = random.choice(ope_index)
                new_tail2 =  tail2[:random_index] + 'C' +tail2[random_index:]
                
                new_lipid_part = combination_lipid(h, tail1,new_tail2)
                q += append_seed(new_lipid_part, update_solution)

            # print(f"### seed_len3: {len(update_solution)}")
            # REMOVE ###
        q = 0
        j = 0
                    
        while j < nRemove :  #tail1 reomve C atom
                h = seed[i,1]
                tail1 =seed[i,2]
                tail2 = seed[i,3]
                ope_index = [m.start() for m in re.finditer('CC', tail1)]
                random_index = random.choice(ope_index)
                new_tail1 =  tail1[:random_index+1]  +tail1[random_index+1:] # kw: bug, this does nothing to the tail!!
                new_lipid_part = combination_lipid(h, new_tail1,tail2)
                j += append_seed(new_lipid_part, update_solution)
        while q < nRemove:   #tail2 reomve C atom
                h = seed[i,1]
                tail1 =seed[i,2]
                tail2 = seed[i,3]
                ope_index = [m.start() for m in re.finditer('CC', tail2)]
                random_index = random.choice(ope_index)
                new_tail2 =  tail2[:random_index+1]  +tail2[random_index+1:] ## kw: bug, this does nothing to the tail!!
                new_lipid_part = combination_lipid(h, tail1,new_tail2)
                q += append_seed(new_lipid_part, update_solution)
        print('finished operation',i)

    return np.asarray(update_solution)




def find_min_index (arr):
    list_a = arr.tolist()
    min_index = list_a.index(min(list_a))
    return min_index
def update_bank(child_solutions, local_opt=False): # kw: local_opt is not used
    cnt_replace = 0
    # o_b = obj_fn(bank[:,:6])
    bank_min = np.min(obj_fn(bank[:,:6]))            #bank
    child_solutions = child_solutions[obj_fn(child_solutions[:,:6]) > bank_min]    
    print('update_child_solutions',child_solutions.shape[0])

    if len(child_solutions) == 0:
        raise PermissionError("child solutions  !")
    
    for i in range(len(child_solutions)):
        fps1 = get_fp(child_solutions[i, 1])
        max_similarity = 0
        max_n = None
        similarity_list =[]
        for _ in range(len(bank)):
            fps2 = get_fp(bank[_, 1])
            dist = TanimotoSimilarity(fps1, fps2)
            similarity_list.append(dist)
            if dist > max_similarity:
                max_similarity = dist
                max_n = _
        # print('mean_similarity',sum(similarity_list)/len(similarity_list))
        if (1 - max_similarity) < dcut:
                if obj_fn(child_solutions[i:i + 1,:6]) > obj_fn(bank[max_n:max_n + 1,:6]):
                    bank[max_n] = child_solutions[i]
                    print('********update************',i)
                    
        else:
                _min = np.argmin(obj_fn(bank[:,:6]))
                if (max_similarity < 0.98) and (obj_fn(bank[_min:_min + 1,:6]) <final_avg.mean()):
            
                # if ((sum(similarity_list)/len(similarity_list)) < 1-dcut) and (obj_fn(bank[_min:_min + 1,:6]) <final_avg.mean()):
                    if obj_fn(child_solutions[i:i + 1,:6]) > obj_fn(bank[_min:_min + 1,:6]):
                        bank[_min] = child_solutions[i]
                        print('********another_update************',i)
                        

        # bank_min = np.min(obj_fn(bank[:,:6])) 
        # if obj_fn(child_solutions[i,:6]) >bank_min:
        #     print('********update************',i)
        #     min_index = find_min_index(obj_fn(bank[:,:6]))
        #     bank[min_index] =child_solutions[i]
        # else:
        #     print('***********no_update*********',i)
    print('update_bank_finished')


    return len(child_solutions)


if __name__ == "__main__":

    target_value = 3
    target_round = args.convergent_round

    R_d = 10**(np.log10(2 / target_value) / int(target_round))

    nbank = args.bank_size  # number of bank conformations
    nseed = args.seed_size  # number of seeds(mating) per iteration
    max_repeat = args.max_round

    total_time = ChkTime()
    chk_load = ChkTime()

    bank = init_bank(
        args.input,
        nbank,
        rseed=args.random_seed,
    )  # 

    chk_load = chk_load.get()

    first_bank = bank   #array

    origin_avg = obj_fn(bank[:6]) #kw: average of only the first 6 molecules?

    plot_list = []



    chk_calc = ChkTime()

    
    # similaity 
    davg = cal_avg_dist(first_bank)
    davg = 1 - davg
    davg = davg * args.dist_coef
    dcut = davg / 2

    # final_avg = origin_avg

    chk_calc = chk_calc.get()

    with open(f"iteration.log", "w") as log_f2:
        log_f2.write(f"load_time: {chk_load:.3f} min\n")
        log_f2.write(f"dist_time: {chk_calc:.1f} min\n")
        log_f2.write(
            f"round  iter  unused  time_seed  time_child  time_update   n_eval\n"
        )

    with open(f"message.log", "w") as log_f:
        log_f.write(f"nbank: {nbank}\n")
        log_f.write(f"nseed: {nseed}\n")
        log_f.write(f"max_repeat: {max_repeat}\n")
        log_f.write(f"R_d: {R_d:.6f} (convergent_round: {target_round})\n")
        # log_f.write(f"D_avg: {davg:.3f} (similarity: {1-davg:.3f})\n")
        # tmp_str = ""
        # for i, j in enumerate(column_name[1:-1]):
        # #     tmp_str += f"{j}: {first_bank[:, i+3].mean():.3f}, "
        # log_f.write(f"init_bank_avg - {tmp_str[:-2]}\n")
        log_f.write(
            f"round   dcut  n_iter  obj_avg  obj_min  obj_max  n_replace  min/round\n"
        )

    save_bank = np.empty([max_repeat, bank.shape[0], 6],dtype=object)

    for round_ in range(max_repeat):
        if (round_ != 0) and (dcut > davg / 3):        # kw: in paper, this value is davg/5
            dcut *= R_d

        timechk = time.time()

        
        niter = 0
        n_replace = 0
        iter_gate = True
        while iter_gate:
            seed = []

            time_seed = ChkTime()
            seed = prepare_seed(bank, seed)       
            time_seed = time_seed.get()

            time_child = ChkTime()
            df_head =pd.read_csv('./head.csv')
            head_list = df_head.iloc[:,0].tolist()
            child_solutions = prepare_child(seed,head_list)               
            child_smi = child_solutions[:,0]
            child_mol = [Chem.MolFromSmiles(s) for s in child_smi]
            child_7 = [True for i in range(len(child_smi))]
            child_solutions = np.column_stack((child_solutions,child_mol,child_7))
            shuffled_index_ = np.random.permutation(
                child_solutions.shape[0])  # @4 에서 추가 됨.
            child_solutions = child_solutions[shuffled_index_]
            time_child = time_child.get()
            

            time_update = ChkTime()
            try:
                # log_f.write(f'## BANK #### @ {np.count_nonzero(bank[:, 2] == True)} #############\n')
                # n_replace += update_bank(child_solutions, True)  # local update
                n_eval = update_bank(child_solutions)  # non-local update
                # n_replace += _n_replace
                # log_f.write(f'## BANK #### @ {np.count_nonzero(bank[:, 2] == True)} #### AFTER ##\n')
            except PermissionError:
                break
            time_update = time_update.get()
            niter += 1
            print('bank_true',np.count_nonzero(bank[:, 7] == True))
            if np.count_nonzero(bank[:, 7] == True) < (nbank - nseed * 0.9):
            # if n_eval < nbank:
                iter_gate = False
            with open(f"iteration.log", "a") as log_f2:
                log_f2.write(
                    f"{round_:>4}  {niter:4}  {np.count_nonzero(bank[:, 7] == True):>6}     {time_seed:6.1f}"
                    f"      {time_child:6.1f}       {time_update:6.1f}        {n_eval}\n"
                )

        final_avg = obj_fn(bank)

        with open(f"message.log", "a") as log_f:
            log_f.write(
                f"{round_:>4}  {dcut:6.3f}   {niter:3}   {final_avg.mean():6.3f}   {final_avg.min():6.3f}   "
                f"{final_avg.max():6.3f}      {(time.time() - timechk)/60:8.2f}\n"
            )

        bank[:, 7] = True  # reset to unused solutions

        plot_list.append(final_avg.mean())
        tmp_bank = np.empty([bank.shape[0],6], dtype=object)
        tmp_bank[:, :6] = bank[:, :6]
        tmp_bank[:, -1] = final_avg
        tmp = np.argsort(tmp_bank[:, -1])
        save_bank[round_] = tmp_bank[tmp[::-1]]

    final_bank = np.empty([bank.shape[0], 6], dtype=object)
    final_bank[:,:6] = bank[:, :6]

    final_bank[:, -1] = final_avg


    print(f"Total Cost Time: {total_time.get():.3f} min")

    np.save(f"list_bank_.npy", save_bank)
    save_smiles = pd.DataFrame(save_bank[:, :, 0])
    save_smiles.to_csv(f"list_smiles.csv",
                       header=False,
                       index=False)
    df = pd.DataFrame(final_bank)
    df.to_csv(f"final_bank.csv", index=False)

    plt.plot(plot_list)
    plt.tight_layout()
    plt.savefig("target_plot.png")
