import numpy as np
from numpy import random
from itertools import product
from itertools import combinations
import math
import copy
import time

hashtable = {}
def permutation_with_replacement(array, elem_num):
    table = [ele for ele in product(array, repeat=elem_num)]

    #print(table)
    return table


def compute_comb_and_check_savehash(coef_domain, vec_indexes, matrix):
    coefs = np.arange(-coef_domain, coef_domain + 1, 1)
    vec_num = len(vec_indexes)

    zeros = np.zeros((1, matrix.shape[1]))[0]
    hashtable[zeros.__str__()] = 1

    coefs_table = permutation_with_replacement(coefs, vec_num)

    for coef in coefs_table:
        # print(hashtable)
        if (np.asarray(coef) == np.zeros((1, vec_num))[0]).all():
            continue

        cnt = 0
        res = np.zeros(matrix.shape[1])

        while cnt < vec_num:
            res += coef[cnt] * matrix[vec_indexes[cnt]]

            cnt += 1

        # if (res == np.array([0, 1, 1])).all():                                          # debug
        #
        #     print(res)
        #     print(hashtable)
        #     print(hashtable.get(res.__str__()))
        # minus_res = -1 * res
        temp = copy.deepcopy(res)
        # print(temp)

        for ele in res:
            if ele != 0:
                ele *= -1

        # print(hashtable)
        #print(res)
        if hashtable.get(res.__str__()) is None:
            # print(True)
            # print("temp is" , temp)
            # print(hashtable.get((-1 * res).__str__()))
            # print(hashtable)
            hashtable[temp.__str__()] = 1
            # print(hashtable)

        else:
            #print(False)
            hashtable.clear()
            print("False: res-vector already exists in hashtable.")
            return False


    # hashtable.clear()
    return True


def compute_comb_and_savehash(coef_domain, vec_indexes, matrix):

    coefs = np.arange(-coef_domain, coef_domain + 1, 1)
    vec_num = len(vec_indexes)
    coefs_table = permutation_with_replacement(coefs, vec_num)

    zeros = np.zeros(matrix.shape[1])
    hashtable[zeros.__str__()] = 1

    for coef in coefs_table:
        if (np.asarray(coef) == np.zeros((1, vec_num))[0]).all():
            continue

        cnt = 0
        res = np.zeros(matrix.shape[1])
        while cnt < vec_num:
            res += coef[cnt] * matrix[vec_indexes[cnt]]
            cnt += 1

        # if (res != zeros).all():
        #     hashtable[res.__str__()] = 1
        # else:
        #     print(coef)
        #     print(vec_indexes)
        #     print(res)
        #     print(zeros)
        #     print("False: Left-res-vector is zeros in case of Non-zero coefficients .")
        #     return False
        if (res == zeros).all():
            # print(coef)
            # print(vec_indexes)
            # print(res)
            # print(zeros)
            print("False: Left-res-vector is zeros in case of Non-zero coefficients .")
            return False
        else:
            if hashtable.get(res.__str__()) is None:
                hashtable[res.__str__()] = []
                hashtable[res.__str__()].append(set(vec_indexes))
            else:
                hashtable[res.__str__()].append(set(vec_indexes))

            # hashtable[res.__str__()] = 1

    return True


def compute_comb_and_checkhash(coef_domain, vec_indexes, matrix):
    coefs = np.arange(-coef_domain, coef_domain + 1, 1)
    vec_num = len(vec_indexes)
    coefs_table = permutation_with_replacement(coefs, vec_num)

    for coef in coefs_table:
        if (np.asarray(coef) == np.zeros((1, vec_num))[0]).all():
            continue

        cnt = 0
        res = np.zeros(matrix.shape[1])
        while cnt < vec_num:
            res += coef[cnt] * matrix[vec_indexes[cnt]]
            cnt += 1


        for ele in res:
            if ele != 0:
                ele *= -1

        if hashtable.get(res.__str__()) is not None:
            # print(coef)
            # print(vec_indexes)
            # print(res)
            # print(hashtable)

            for vec_idx_set in hashtable[res.__str__()]:
                if vec_idx_set & set(vec_indexes):
                    return True
                else:
                    print("False: -1*right-res-vector already exists in hashtable.")
                    return False

        if (res == np.zeros(matrix.shape[1])).all():
            print("False: right-res-vector is zeros in case of non-zero coefficients.")
            return False
    # hashtable.clear()

    return True


def check_kruskal_rank(kr_rank, matrix, is_binary=True):
    if kr_rank % 2 == 0:
        index_array = range(0, matrix.shape[0])
        vec_table = combinations(index_array, int(kr_rank / 2))

        for vec_comb in vec_table:
            if compute_comb_and_check_savehash(math.factorial(kr_rank), vec_comb, matrix) == False:
                print("This matrix's kruskal rank is less than %d." % kr_rank)
                hashtable.clear()
                return

        print("This matrix's kruskal rank is at least %d." % kr_rank)

    if kr_rank % 2 == 1:
        index_array = range(0, matrix.shape[0])
        vec_table_left = combinations(index_array, int((kr_rank + 1) / 2))

        for vec_comb_left in vec_table_left:
            if compute_comb_and_savehash(math.factorial(kr_rank), vec_comb_left, matrix) == False:
                print("This matrix's kruskal rank is less than %d." % kr_rank)
                hashtable.clear()
                return

        vec_table_right = combinations(index_array, int((kr_rank - 1) / 2))

        for vec_comb_right in vec_table_right:
            if compute_comb_and_checkhash(math.factorial(kr_rank), vec_comb_right, matrix) == False:
                print("This matrix's kruskal rank is less than %d." % kr_rank)
                hashtable.clear()
                return

        print("This matrix's kruskal rank is at least %d." % kr_rank)

    hashtable.clear()
    return



# mat1 = [[0, 0, 1],
#         [0, 1, 0],
#         [1, 1, 0]]
# mat1 = np.asarray(mat1)
# print(mat1)
# mat1 = mat1.T

time_used = []
for i in range(10):
    mat = random.randint(0, 2, (10, 10))
    #print(mat)
    mat = mat.T
    start_time = time.process_time()

    check_kruskal_rank(2, mat)
    # check_kruskal_rank(3, mat)
    end_time = time.process_time()

    print('runtime: %s s' % ((end_time - start_time)))
    time_used.append(end_time - start_time)

time_used = np.asarray(time_used)
print(time_used.mean())





