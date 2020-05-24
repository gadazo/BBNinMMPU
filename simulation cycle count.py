import math
import matplotlib.pyplot as plt
import pandas as pd

space_time_var = {
    'xnor': {
        'time_oriented': {
            'time': 4,
            'space': 4,
            'out_bits': 1,
            'init': 0,
        },
        'space_oriented': {
            'time': 5,
            'space': 3,
            'out_bits': 1,
            'init': 1,
        }
    },
    'half_adder': {
        'time_oriented': {
            'time': 5,
            'space': 5,
            'out_bits': 2,
            'init': 0,
        },
        'space_oriented': {
            'time': 7,
            'space': 3,
            'out_bits': 2,
            'init': 2,
        }
    },
    'full_adder': {
        'time_oriented': {
            'time': 9,
            'space': 9,
            'out_bits': 2,
            'init': 0,
        },
        'space_oriented': {
            'time': 14,
            'space': 4,
            'out_bits': 2,
            'init': 5,
        }
    }
}

VGG16_program = [
    {
        'type': 'convolution',
        'Kw': 3,
        'Kh': 3,
        'Cout': 64,
        'stride': 1,
        'padding': 1,
    },
    {
        'type': 'convolution',
        'Kw': 3,
        'Kh': 3,
        'Cout': 64,
        'stride': 1,
        'padding': 1,
    },
    {
        'type': 'max_pooling',
        'Kw': 2,
        'Kh': 2,
        'Cout': 64,
        'stride': 2,
        'padding': 0,
    },
]

simulation_vars = {
    'init_one_bit_power': 1,
    'init_per_bit_power': 1,
    '2bit_nor_power': 2,
    '1bit_inv_power': 1,
    'nbit_nor_power': 1,
}


simulation_vars = {
    'latency': {
        'read_to_bank_io' : 20,
        'write_between_mats': 30,  # moving data between mats in the same array
        'move_between_banks': 20,
        'move_between_chip': 1000,
        'move_between_ranks': 3000,
        'write_from_bank_io': 30,
        },

    'power':
        {
        'read_to_bank_io': 20,
        'write_between_mats': 30,  # moving data between mats in the same array
        'move_between_banks': 20,
        'move_between_chip': 1000,
        'move_between_ranks': 3000,
        'write_from_bank_io': 30,
        'write_per_bit': 2,
        'init_one_bit': 1,
        'init_per_bit': 1,
        '2bit_nor': 2,
        '1bit_inv': 1,
        'nbit_nor': 1,
        }
}

structure_vars = {
    'mats_in_subarray': 16,
    'subarrays_in_bank': 64,
    'banks_in_chips': 8,
    'rows_in_mat': 1024,
    'read_bandwidth' : 16
}

simulation_flags = {
    'multiple_writes': True,
    'Read_style': 'row', # choises : row / col
    'after_action_order': 'no_order', # choices : no_order / matrices / write_oriented
    'orientation' : 'space_oriented', # choices : space_oriented / time_oriented
    'popcount_method': 'incrementor',  # choices : incrementor / tree_counter
    'find_best_mat': True
}

input_vars = {
    'Win': 224,
    'Hin': 224,
    'Cin': 3,
}





# returns the power of one complex gate
def power_gate(simulation_vars, space_time_var, orientation, gate_type):
    nor_actions = space_time_var[gate_type][orientation]['time'] - space_time_var[gate_type][orientation]['init']
    bits_init = space_time_var[gate_type][orientation]['init']
    after_action_init = space_time_var[gate_type][orientation]['space'] + \
                        space_time_var[gate_type][orientation]['out_bits']  # num of space and one outcome space
    power_needed = nor_actions * simulation_vars['2bit_nor_power']
    power_needed += bits_init * simulation_vars['init_one_bit_power']
    power_needed += after_action_init * simulation_vars['init_per_bit_power']
    return power_needed

def space_action(simulation_vars, space_time_var, kernel_num_bit, sim_flags):
    # how much the input and the extra space needed for a specific action in a row
    space_needed = kernel_num_bit
    space_needed += 1  # first output
    if sim_flags['popcount_method'] == 'incrementor':
        init_num = math.ceil((pow(2, math.ceil(math.log(kernel_num_bit, 2))) - kernel_num_bit - 1) / 2)
        if init_num > 0:
            space_needed += math.ceil(math.log(init_num, 2) + 1)  # extra space for the intermediate outputs
        else:
            space_needed += 2
    elif sim_flags['popcount_method'] == 'tree_counter':
        space_needed += (math.ceil(kernel_num_bit / 3)) * 2  # for the first adding
    return space_needed

# calculating the number of space and time needed for one action
# The function assumes the data and kernel is already in place
def time_power_action(simulation_vars, space_time_var, kernel_num_bit, sim_flags):
    time_needed = 0
    power_needed = 0
    # need tha data and corresponding kernel bit and xnor outcome
    time_needed += kernel_num_bit * space_time_var['xnor'][sim_flags['orientation']]['time']
    power_needed += kernel_num_bit * power_gate(simulation_vars, space_time_var, sim_flags['orientation'], 'xnor')
    if sim_flags['popcount_method'] == 'incrementor':
        for p in range(1, kernel_num_bit + 1):
            # added 1 below because for every new half adder we need to override space so one init
            time_needed += (1 + space_time_var['half_adder'][sim_flags['orientation']]['time']) * math.ceil(math.log(p, 2) + 1)
            power_needed += math.ceil(math.log(p, 2) + 1) * \
                            power_gate(simulation_vars, space_time_var, sim_flags['orientation'], 'half_adder')
    elif sim_flags['popcount_method'] == 'tree_counter':
        for p in range(math.ceil(math.log(kernel_num_bit, 2))):
            # added 1 below beacuse for every new half adder we need to override space so one init
            time_needed += space_time_var['full_adder'][sim_flags['orientation']]['time'] * (1 + p) * \
                           math.floor(kernel_num_bit / (3 * math.pow(2, p)))
            power_needed += (1 + p) * math.floor(kernel_num_bit / (3 * math.pow(2, p))) * \
                            power_gate(simulation_vars, space_time_var, sim_flags['orientation'], 'full_adder')
    return time_needed, power_needed


def max_bit_in_mat(str_vars, kernel_num_bit, sim_vars,sim_flags):
    _space = space_action(sim_vars, space_time_var, kernel_num_bit,sim_flags)
    available_bits = str_vars['rows_in_mat'] - kernel_num_bit
    max_bits_int_row = math.floor(available_bits / _space)
    return max_bits_int_row * str_vars['rows_in_mat']

def simulation(struct_vars, program_list, s_t_var, sim_vars, input_vars,sim_flags):
    input_stats = input_vars
    sum_power = 0
    sum_cycles = 0
    for stage in program_list:
        # the following program calculates the time and power to reorder the data for the next action
        temp_time, temp_power = reorganize_data(input_vars, stage, struct_vars,s_t_var,sim_flags)


def after_action(input_vars, stage, struct_vars, s_t_var, sim_flags) -> (int, int):
    if sim_flags['after_action_order'] == 'no_order':
        return 0,0
    elif sim_flags['after_action_order'] == 'matrices':
        return 2,2
    else:
        return 1,1

############################# Graphs
a = 2
b = 32
linspace = range(a, b)
space = []
n = b - a
space.append([0] * n)
space.append([0] * n)
space.append([0] * n)
space.append([0] * n)
time = []
time.append([0] * n)
time.append([0] * n)
time.append([0] * n)
time.append([0] * n)
power = []
power.append([0] * n)
power.append([0] * n)
power.append([0] * n)
power.append([0] * n)
max_bit_in_mat = []
max_bit_in_mat.append([0] * n)
max_bit_in_mat.append([0] * n)
max_bit_in_mat.append([0] * n)
max_bit_in_mat.append([0] * n)
max_bit_in_mat.append([0] * n)
max_bit_in_mat.append([0] * n)
max_bit_in_mat.append([0] * n)
max_bit_in_mat.append([0] * n)
for p in linspace:
    space[0][p - 2], time[0][p - 2], power[0][p - 2] = spaceTimePowerAction(simulation_vars, space_time_var, p,
                                                                            'incrementor', 'time_oriented')
    space[1][p - 2], time[1][p - 2], power[1][p - 2] = spaceTimePowerAction(simulation_vars, space_time_var, p,
                                                                            'incrementor', 'space_oriented')
    space[2][p - 2], time[2][p - 2], power[2][p - 2] = spaceTimePowerAction(simulation_vars, space_time_var, p,
                                                                            'tree_counter', 'time_oriented')
    space[3][p - 2], time[3][p - 2], power[3][p - 2] = spaceTimePowerAction(simulation_vars, space_time_var, p,
                                                                            'tree_counter', 'space_oriented')
    max_bit_in_mat[0][p - 2] = maxBitInMat(simulation_vars, 512, p, 'incrementor', 'time_oriented')
    max_bit_in_mat[1][p - 2] = maxBitInMat(simulation_vars, 512, p, 'incrementor', 'space_oriented')
    max_bit_in_mat[2][p - 2] = maxBitInMat(simulation_vars, 512, p, 'tree_counter', 'time_oriented')
    max_bit_in_mat[3][p - 2] = maxBitInMat(simulation_vars, 512, p, 'tree_counter', 'space_oriented')
    max_bit_in_mat[4][p - 2] = maxBitInMat(simulation_vars, 1024, p, 'incrementor', 'time_oriented')
    max_bit_in_mat[5][p - 2] = maxBitInMat(simulation_vars, 1024, p, 'incrementor', 'space_oriented')
    max_bit_in_mat[6][p - 2] = maxBitInMat(simulation_vars, 1024, p, 'tree_counter', 'time_oriented')
    max_bit_in_mat[7][p - 2] = maxBitInMat(simulation_vars, 1024, p, 'tree_counter', 'space_oriented')

fig, (ax1, ax2) = plt.subplots(2, 2)
ax1[0].plot(linspace, space[0])
ax1[0].plot(linspace, space[1])
ax1[0].plot(linspace, space[2])
ax1[0].plot(linspace, space[3])
ax1[0].set_title('Space of a single action on N bit')
ax1[0].legend(['incr time', 'incr space', 'tree time', 'tree space'], loc='upper left')
ax1[1].plot(linspace, power[0])
ax1[1].plot(linspace, power[1])
ax1[1].plot(linspace, power[2])
ax1[1].plot(linspace, power[3])
ax1[1].set_title('Power of a single action on N bit')
ax1[1].legend(['incr time', 'incr space', 'tree time', 'tree space'], loc='upper left')
ax2[0].plot(linspace, time[0])
ax2[0].plot(linspace, time[1])
ax2[0].plot(linspace, time[2])
ax2[0].plot(linspace, time[3])
ax2[0].set_title('Cycles of a single action on N bit')
ax2[0].legend(['incr time', 'incr space', 'tree time', 'tree space'], loc='upper left')
ax2[1].plot(linspace, max_bit_in_mat[0])
ax2[1].plot(linspace, max_bit_in_mat[1])
ax2[1].plot(linspace, max_bit_in_mat[2])
ax2[1].plot(linspace, max_bit_in_mat[3])
ax2[1].plot(linspace, max_bit_in_mat[4])
ax2[1].plot(linspace, max_bit_in_mat[5])
ax2[1].plot(linspace, max_bit_in_mat[6])
ax2[1].plot(linspace, max_bit_in_mat[7])
ax2[1].set_title('max bits in mat')
ax2[1].legend(['incr time (512)', 'incr space (512)', 'tree time (512)', 'tree space (512)',
               'incr time (1024)', 'incr space (1024)', 'tree time (1024)', 'tree space (1024)'], loc='upper right')

plt.show()
