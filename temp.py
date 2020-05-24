import math
import numpy as np

def increment_count_with_decision(data_num_bits):
    len = data_num_bits
    available_space = 0
    added_space = 0
    result_len = math.floor(math.log(len, 2)) + 1
    init_num = math.ceil((pow(2, math.ceil(math.log(len, 2))) - len - 1) / 2)  # for decision
    print(init_num)
    if (init_num == 0 & (not math.log2(len).is_integer())):
        init_num = 1
    init_num_of_bits = (math.floor(math.log(init_num, 2)) + 1)
    num_of_adders = init_num_of_bits + 1
    available_space -= (int(init_num/2))*(init_num > 0)
    for i in range(len):
        curr_num_of_bits = math.floor(math.log(i + 1, 2)) + 1

        if (curr_num_of_bits == init_num_of_bits):
            num_of_bits = curr_num_of_bits
            num_of_adders = curr_num_of_bits + 1
            init_num_of_bits = 0
        elif (init_num_of_bits != curr_num_of_bits and num_of_adders != result_len):
            num_of_adders = num_of_adders + 1
            num_of_bits = curr_num_of_bits

        #result[0], carry, cycle_count = bool_halfadder_2outputs(result[0], vec[i], cycle_count, minimal)
        #uses one data bit and one result bit outputs 2 bit
        if available_space >= 2:
            available_space += 1
        else:
            added_space += 2-available_space
            available_space += 2
        for j in range(num_of_adders - 1):
            #result[j + 1], carry, cycle_count = bool_halfadder_2outputs(result[j + 1], carry, cycle_count,
            #                                                           minimal)
            print("avs: "+str(available_space))
            print("ads: "+str(added_space))
            if available_space < 2:
                added_space += 2 - available_space
                available_space += 2

    return added_space
print(increment_count_with_decision(28))