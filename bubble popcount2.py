import numpy as np
import math

DEBUG = 1
p = 19

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

def bool_fulladder_2outputs(a, b, c, nor_count):
    res_xor_ab, res_nor_ab, nor_count = bool_xnor_2outputs(a, b, nor_count)
    res_sum, res_nor_c_xorab, nor_count = bool_xnor_2outputs(res_xor_ab, c, nor_count)
    res_cout, nor_count = bool_nor(res_nor_ab, res_nor_c_xorab, nor_count)
    return res_sum, res_cout, nor_count

def bool_halfadder_2outputs(a, b, nor_count):
    res_and_ab, nor_count = bool_and(a, b, nor_count)
    res_nor_ab, nor_count = bool_nor(a, b, nor_count)
    res_sum, nor_count = bool_nor(res_and_ab, res_nor_ab, nor_count)
    return res_sum, res_and_ab, nor_count


def bool_adder_nbit(a_vec, b_vec, out_vec, n, start_a, start_b, start_out, nor_count):
    c = np.zeros(2)
    current_c = 0
    # first adder- only half adder
    out_vec[start_out], c[current_c], nor_count = bool_halfadder_2outputs(a_vec[start_a], b_vec[start_b], nor_count)

    # middle adder - full adder, each residue will be input for the next
    for j in range(1, n-1):
        out_vec[start_out + j], c[1-current_c], nor_count = \
            bool_fulladder_2outputs(a_vec[j + start_a], b_vec[j + start_b], c[current_c], nor_count)
        current_c = 1-current_c

    # last adder - full adder, residue will be in the last bit of the output
    out_vec[start_out + n-1], out_vec[start_out + n], nor_count = \
            bool_fulladder_2outputs(a_vec[n-1 + start_a], b_vec[n-1 + start_b], c[current_c], nor_count)


def bool_adder_nbit_2diff(odd_vec, b_vec, out_vec, n, start_a, start_b, start_out, cycle_count):
    c = np.zeros(2)
    current_c = 0
    # first adder- only half adder
    out_vec[start_out], c[current_c], cycle_count = bool_halfadder_2outputs(odd_vec[start_a],
                                                                            b_vec[start_b], cycle_count)

    # middle adder - full adder, each residue will be input for the next
    for j in range(1, n - 2):
        out_vec[start_out + j], c[1 - current_c], nor_count = \
            bool_fulladder_2outputs(odd_vec[j + start_a], b_vec[j + start_b], c[current_c], cycle_count)
        current_c = 1 - current_c

    out_vec[n-2], c[1-current_c], cycle_count = bool_halfadder_2outputs(odd_vec[n-2], c[current_c], cycle_count)
    out_vec[n-1], cycle_count = bool_or(c[1-current_c], odd_vec[n-2], cycle_count)


def bool_adder_nbit_1diff(big_vec, small_vec, out_vec, n, start_a, start_b, start_out, cycle_count):
    c = np.zeros(2)
    current_c = 0
    # first adder- only half adder
    out_vec[start_out], c[current_c], cycle_count = bool_halfadder_2outputs(big_vec[start_a],
                                                                            small_vec[start_b], cycle_count)

    # middle adder - full adder, each residue will be input for the next
    for j in range(1, n-1):
        out_vec[start_out + j], c[1 - current_c], nor_count = \
            bool_fulladder_2outputs(big_vec[j + start_a], small_vec[j + start_b], c[current_c], cycle_count)
        current_c = 1 - current_c

    out_vec[n-1], out_vec[n], cycle_count = bool_halfadder_2outputs(big_vec[n - 1], c[current_c], cycle_count)


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
                temp_vec[j+1], nor_count = bool_and(vec[j], vec[j+1], nor_count)
                temp_vec[j], nor_count = bool_or(vec[j], vec[j+1], nor_count)
            else:
                break
        vec = np.copy(temp_vec)
    return vec, nor_count


def debug_print(*str):
    if DEBUG:
        print(*str)


# Tree adder ones count:
# Input: Nbit vector
# Output: binary representation of ones count
# Algorithm:
# 1. summation of the ones in the vector to 2bit representation (Full-adder = 3bit-->2bit)
# 2. Adding pairs together until there is uneven number of summations
# 3. Adding together all the possible pairs (Mbit --> (M+1)bit)
# 4. Adding the the first (M+1)bit pair together and adding the left
#                           summation to them(Mbit) ((M+1)bit + (M+1)bit + Mbit --> (M+2)bit
# 5. from now on there will be only pairs . so will add pairs together until one summation left
def count_ones_vec(vec, cycle_count):
    len = np.size(vec)
    len_remain = len
    current_vec = 0
    temp_vec = np.zeros((2, (math.ceil(len/3))*2))  # FIXME: need to add cycle_count and maybe less space

    # first step : summation of the ones in the vector to 2bit representation
    for i in range(math.ceil(len/3)):
        if len_remain >= 3:
            temp_vec[0][i*2], temp_vec[0][i*2+1], cycle_count = bool_fulladder_2outputs(vec[i * 3], vec[i * 3 + 1], vec[i * 3 + 2], cycle_count)
        elif len_remain % 3 == 2:
            temp_vec[0][i*2], temp_vec[0][i*2+1], cycle_count = bool_halfadder_2outputs(vec[i * 3], vec[i * 3 + 1], cycle_count)
        else:  # adding the last remained bit
            temp_vec[0][i*2] = vec[i*3]
            cycle_count += 2  # moving bit takes two cycles
        len_remain -= 3

    debug_print("debug: after 2bit\n", temp_vec[0])

    odd_flag = 0
    last_n_bit = 0
    # second stage : Adding pairs together until there is uneven number of summations
    number_of_summations = math.ceil(len / 3)  # number of 2bits
    num_rep_bit = int(math.ceil(math.log(len, 2)))  # the number needed to represent the number of added bits
    if math.log(len, 2) % 1 == 0:
        num_rep_bit += 1  # add 1 if it is a power of 2
    debug_print("debug: num_rep_bit", num_rep_bit)
    for n_bit in range(2, num_rep_bit):
        if number_of_summations % 2 != 0:
            odd_flag = 1
            last_n_bit = n_bit
            break
        for i in range(int(number_of_summations/2)):
            bool_adder_nbit(temp_vec[current_vec], temp_vec[current_vec], temp_vec[1-current_vec],
                            n_bit, i * n_bit * 2, i * n_bit * 2 + n_bit, i * (n_bit+1), cycle_count)
        current_vec = 1 - current_vec
        #print("debug1: after ", n_bit, "bit\n", temp_vec[current_vec])
        number_of_summations = math.ceil(number_of_summations / 2)
        #print("debug: number_of_summations", number_of_summations)

    if odd_flag:
        # Third stage: adding together all the possible pairs ( Mbit --> (M+1)bit )
        n_bit = last_n_bit
        for i in range(math.floor(number_of_summations/2)):
            bool_adder_nbit(temp_vec[current_vec], temp_vec[current_vec], temp_vec[1-current_vec],
                            n_bit, i * n_bit * 2,
                            (i * n_bit * 2) + n_bit, i * (n_bit + 1), cycle_count)
        current_vec = 1 - current_vec
        debug_print("debug2: after ", n_bit, "bit\n", temp_vec[current_vec])
        n_bit += 1


        # Fourth stage: Adding the the first (M+1)bit pair together
        if number_of_summations != 3:
            temp_odd_vec = np.zeros(last_n_bit+2)
            bool_adder_nbit(temp_vec[current_vec], temp_vec[current_vec], temp_odd_vec,
                            n_bit, 0, n_bit, 0, cycle_count)  # the first two (M+1)bit to one (M+2)bit
            debug_print("debug3: after", n_bit, "bit\n", temp_vec[current_vec])
            n_bit += 1

            # adding the left Mbit to the (M+2)bit
            debug_print("debug4: num of sum: ", number_of_summations)
            bool_adder_nbit_2diff(temp_odd_vec, temp_vec[1 - current_vec], temp_vec[1 - current_vec],
                                  n_bit, 0, (number_of_summations-1) * (n_bit-2), 0, cycle_count)
        else:
            bool_adder_nbit_1diff(temp_vec[current_vec], temp_vec[1 - current_vec], temp_vec[1 - current_vec],
                                  n_bit, 0, 2 * (n_bit - 1), 0, cycle_count)
            n_bit += 1

        number_of_summations = math.ceil(number_of_summations / 2)
        debug_print("debug5: num of sum: ", number_of_summations)
        #   adding the rest of pairs ((M+1)bit --> (M+2)bit)
        for i in range(1, math.ceil(number_of_summations/2)):
            bool_adder_nbit(temp_vec[current_vec], temp_vec[current_vec], temp_vec[1-current_vec],
                            (n_bit-1), (2 * i * (n_bit-1))-(n_bit-1),
                            i * (n_bit-1) * 2, i * n_bit, cycle_count)
        current_vec = 1 - current_vec
        debug_print("debug6: after", n_bit, "bit\n", temp_vec[current_vec])

        # from now on we assured that there will be only pairs to sum together
        for n_bit2 in range(n_bit, num_rep_bit):
            number_of_summations = math.floor(number_of_summations / 2)
            for i in range(int(number_of_summations/2)):
                bool_adder_nbit(temp_vec[current_vec], temp_vec[current_vec], temp_vec[1-current_vec],
                                n_bit2, i * (n_bit2)*2, (i * (n_bit2) * 2) + (n_bit2), i * (n_bit2+1), cycle_count)
            current_vec = 1-current_vec
            debug_print("debug7: after", n_bit2+1, "bit\n", temp_vec[current_vec])

    return temp_vec[current_vec][range(int(math.ceil(math.log(len+1, 2))))], cycle_count

def vec_to_int(vec):
    n = np.size(vec)
    sum1 = 0
    j = 1
    for i in range(0, n):
        sum1 += vec[i]*j
        j *= 2
    return sum1


if DEBUG:
    print(np.ones(p))
    print(count_ones_vec(np.ones(p), 0))
else:
    for i in range(2, p+1):
        vec, cycle_count = count_ones_vec(np.ones(i), 0)
        if vec_to_int(vec) == i:
            print("win: ", vec, cycle_count)
        else:
            print("lose", p)

# arr = bool_vector_gen(p)
# for j in range(pow(2, p)):
#     print(arr[j])
#     print(count_ones_vec(arr[j], 0))
#     print()



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


# po = 4
## checks if the theory of adding 0 and 1 to compliment the closest power of 2 works
# arr = bool_vector_gen(po)
# count = 0
#
# for i in range(pow(2, po)):
#     print(math.ceil((pow(2, math.ceil(math.log(po, 2))) - po-1)/2))
#     print(pow(2, math.ceil(math.log(po, 2)))/2)
#     one_count = ones_count_vec(arr[i])
#     one_count_mod = one_count + math.ceil((pow(2, math.ceil(math.log(po, 2))) - po-1)/2)
#     if one_count >= po/2 and one_count_mod >= pow(2, math.ceil(math.log(po, 2)))/2:
#         pass
#     elif one_count < po/2 and one_count_mod < (pow(2, math.ceil(math.log(po, 2)))-1)/2:
#         pass
#     else:
#         count += 1
# if count > 0:
#   print("nooooo")
# else:
#   print("horrray")
# ~~~~~~~~~~~ nice result: is when you  need to add power of 2 bits the bit that you need to check is the MSB-1 or MSB
