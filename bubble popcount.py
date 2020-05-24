import numpy as np
import math


def bool_init1(nor_count):
    return 1, nor_count+1

def bool_nor(a, b, nor_count):
    if a or b:
        return False, nor_count+1
    return True, nor_count+1


def bool_and(a, b, nor_count):
    res_not_a, nor_count = bool_nor(a, a, nor_count)
    res_not_b, nor_count = bool_nor(b, b, nor_count)
    return bool_nor(res_not_a, res_not_b, nor_count)


def bool_or(a, b, nor_count):
    res_nor, nor_count = bool_nor(a, b, nor_count)
    return bool_nor(res_nor, res_nor, nor_count)

def bool_xor(a, b, nor_count):
    res_nor_ab_c1, nor_count = bool_nor(a, b, nor_count)
    res_nor_bc1_c2, nor_count = bool_nor(b, res_nor_ab_c1, nor_count)
    res_nor_ac1_c3, nor_count = bool_nor(a, res_nor_ab_c1,nor_count)
    res_xnor, nor_count = bool_nor(res_nor_bc1_c2, res_nor_ac1_c3, nor_count)
    res_xor, nor_count = bool_nor(res_xnor, res_xnor, nor_count)
    return res_xor, nor_count

def bool_xnor_2outputs(a, b, nor_count):
    res_nor_ab_c1, nor_count = bool_nor(a, b, nor_count)
    res_nor_bc1_c2, nor_count = bool_nor(b, res_nor_ab_c1, nor_count)
    res_nor_ac1_c3, nor_count = bool_nor(a, res_nor_ab_c1,nor_count)
    res_xor, nor_count = bool_nor(res_nor_bc1_c2, res_nor_ac1_c3, nor_count)
    return res_xor, res_nor_ab_c1, nor_count


# 9 cycles
def bool_fulladder_2outputs(a, b, c, nor_count):
    res_xor_ab, res_nor_ab, nor_count = bool_xnor_2outputs(a, b, nor_count)
    res_sum, res_nor_c_xorab, nor_count = bool_xnor_2outputs(res_xor_ab, c, nor_count)
    res_cout, nor_count = bool_nor(res_nor_ab, res_nor_c_xorab, nor_count)
    return res_sum, res_cout, nor_count


# 5_cycles
def bool_halfadder_2outputs(a, b, nor_count):
    res_and_ab, nor_count = bool_and(a, b, nor_count)
    res_nor_ab, nor_count = bool_nor(a, b, nor_count)
    res_sum, nor_count = bool_nor(res_and_ab, res_nor_ab, nor_count)
    return res_sum, res_and_ab, nor_count


def bool_adder_nbit(a_vec, b_vec, out_vec, n, start_a, start_b, start_out, nor_count):
    c = 0
    # first adder- only half adder
    out_vec[start_out], c, nor_count = bool_halfadder_2outputs(a_vec[start_a], b_vec[start_b], nor_count)

    # middle adder - full adder, each residue will be input for the next
    for j in range(1, n-1):
        out_vec[start_out + j], c, nor_count = \
            bool_fulladder_2outputs(a_vec[j + start_a], b_vec[j + start_b], c, nor_count)

    # last adder - full adder, residue will be in the last bit of the output
    out_vec[start_out + n-1], out_vec[start_out + n], nor_count = \
            bool_fulladder_2outputs(a_vec[n-1 + start_a], b_vec[n-1 + start_b], c, nor_count)


# cycle count : 5
def bool_switch(a,b,cycle_count):
    smaller, cycle_count = bool_and(a,b,cycle_count)
    bigger, cycle_count = bool_or(a,b,cycle_count)
    return smaller, bigger, cycle_count


def bool_vector_gen(n):
    arr = np.zeros((pow(2, n), n))
    for i in range(pow(2, n)):
        num = int(np.binary_repr(i, n))
        for j in range(n):
            arr[i][j] = num % 10
            num = int(num / 10)
    return arr


def bubble_ones_vec(vec):
    nor_count = 0
    for i in range(np.size(vec)):
        temp_vec = np.copy(vec)
        if i % 2 == 0:
            is_odd = 0
        else:
            is_odd = 1
        for j in range(is_odd, np.size(vec), 2):
            if j + 1 < np.size(vec):
                temp_vec[j+1], temp_vec[j], nor_count = bool_switch(vec[j], vec[j+1], nor_count)
            else:
                break
        vec = np.copy(temp_vec)
    return vec, nor_count


def count_ones_vec(vec, cycle_count):
    len = np.size(vec)
    # at each iteration of the summation the needed space is reduced
    # the largest space which needed is for the first iteration.
    # on the first iteration for each 3 bits we will get 2 bits
    # so the space needed is ceil(length/3)*2
    temp_vec = np.zeros((2, (math.ceil(len/3))*2)) # FIXME: need to add cycle_count
    len_remain = len
    current_vec = 0
    # first iteration (from ones to counting numbers)
    for i in range(math.ceil(len/3)):
        if len_remain >= 3:
            temp_vec[0][i*2], temp_vec[0][i*2+1], cycle_count = bool_fulladder_2outputs(vec[i * 3], vec[i * 3 + 1], vec[i * 3 + 2], cycle_count)
        elif len_remain % 3 == 2:
            temp_vec[0][i*2], temp_vec[0][i*2+1], cycle_count = bool_halfadder_2outputs(vec[i * 3], vec[i * 3 + 1], cycle_count)
        else: # adding the last remained bit
            temp_vec[0][i*2] = vec[i*3]
            cycle_count += 2  # moving bit takes two cycles
        len_remain -= 3
    #print("after 2bit\n", temp_vec[0])
    # second and third iteration (2bit-->3bit, 3bit-->4bit)
    if len > 3:
        # adding all pairs (will left the last 2bit when the number of 2bits are odd)
        # adding 2 2bits number will result 3bit number
        current_vec = 0
        for i in range(math.floor(math.ceil(len/3)/2)):
            bool_adder_nbit(temp_vec[0], temp_vec[0], temp_vec[1], 2, i * 4, i * 4 + 2, i * 3, cycle_count)
        # if there were left 2bit number from the first loop will be added now to the first 4bit number
        # choose first for convenience , last will be better for physical implementation
        #print("after 3bit\n", temp_vec[1])
        if math.ceil(len / 3) % 2 != 0:
            temp_odd_vec = np.zeros(4)
            bool_adder_nbit(temp_vec[1], temp_vec[1], temp_odd_vec, 3, 0, 3, 0, cycle_count) #the first two 3bit to one 4bit
            #print("temp vec\n", temp_odd_vec)
            #print(" debug 3:", math.floor(math.floor(math.ceil(len/3)/2)/2))
            for i in range(2, math.floor(math.floor(math.ceil(len/3)/2)/2)):
                bool_adder_nbit(temp_vec[1], temp_vec[1], temp_vec[0], 3, i * 6, (i * 6) + 3, (i-2) * 4, cycle_count)
            #print("debug4: ", math.ceil(math.floor(len/3)*2))
            temp_vec[0][0], c0, cycle_count = \
                bool_halfadder_2outputs(temp_odd_vec[0], temp_vec[0][math.floor((len-1)/3)*2], cycle_count)
            temp_vec[0][1], c1, cycle_count = \
                    bool_fulladder_2outputs(temp_odd_vec[1], temp_vec[0][math.floor((len-1)/3)*2+1], c0, cycle_count)
            temp_vec[0][2], c0, cycle_count = bool_halfadder_2outputs(temp_odd_vec[2], c1, cycle_count)
            temp_vec[0][3], cycle_count = bool_or(c0, temp_odd_vec[2], cycle_count)

        else:
            #print("here")
            #print("debug2: ", math.floor(math.ceil(math.ceil(len/3)/2)/2))
            current_vec = 1
            for i in range(math.floor(math.ceil(math.ceil(len/3)/2)/2)):
                bool_adder_nbit(temp_vec[1], temp_vec[1], temp_vec[0], 3, i * 6, (i * 6) + 3, i * 4, cycle_count)
                current_vec = 0


    number_of_pairs = math.floor(math.ceil(math.ceil(math.ceil(len/3)/2)/2))
    # from now on we assured that there will be only pairs to sum together
    for n_bit in range(5, int(math.ceil(math.log(len, 2)))+1):
        number_of_pairs = math.floor(number_of_pairs / 2)
        for i in range(number_of_pairs):
            bool_adder_nbit(temp_vec[current_vec], temp_vec[current_vec], temp_vec[1-current_vec],
                            n_bit-1, i * (n_bit-1)*2, (i * (n_bit-1) * 2) + (n_bit-1), i * n_bit, cycle_count)
        current_vec = 1-current_vec

    return temp_vec[current_vec][range(int(math.ceil(math.log(len+1, 2))))], cycle_count


p = 15

arr = bool_vector_gen(p)
for j in range(pow(2, p)):
    print(arr[j])
    print(count_ones_vec(arr[j], 0))
    print()


## here is the check for the bubble popcount
## it checks the cycle count( how many nors) from 2bit to p
## if the bubble popcount ( bubble_ones_vec) gives the wrong answer it will bring the cycle count to -1

# p = 15
# cycle_cnt_vec = np.zeros(p-1)
# for n in range(2, p+1):
#     arr = bool_vector_gen(n)
#     wrong_cnt = 0
#     cycle_cnt = 0
#     for k in range(pow(2, n)):
#         vec, cycle_cnt = bubble_ones_vec(arr[k])
#         kop = 0
#         for num in arr[k]:
#             if num:
#                 kop += 1
#         if (vec[math.ceil(n/2)-1] and kop < n/2) or (not(vec[math.ceil(n/2)-1]) and kop >= n/2):
#             wrong_cnt += 1
#     if wrong_cnt:
#         cycle_cnt = -1
#     cycle_cnt_vec[n-2] = cycle_cnt
#
# print(cycle_cnt_vec)


def ones_count_vec(vec):
    count = 0
    for i in range(np.size(vec)):
        if vec[i]:
            count += 1
    return count



## checks if the theory of adding 0 and 1 to compliment the closest power of 2 works
# arr = bool_vector_gen(po)
# for i in range(pow(2, po)):
#     print((pow(2, math.ceil(math.log(po, 2))) - po-1))
#     one_count = ones_count_vec(arr[i])
#     one_count_mod = one_count + math.ceil((pow(2, math.ceil(math.log(po, 2))) - po-1)/2)
#     if one_count >= po/2 and one_count_mod >= pow(2, math.ceil(math.log(po, 2)))/2:
#         print()
#     elif one_count < po/2 and one_count_mod < (pow(2, math.ceil(math.log(po, 2)))-1)/2:
#         print("horray!")
#     else:
#         print("noooo")
