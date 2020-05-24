from math import floor
from math import ceil
from math import log
import pandas as pd


def calculate_needed_bits(input_vars, stage, bit_id):
    needed_data_bit_list = []
    needed_kernel_bit_list = []
    if stage['type'] == 'convolution':
        """
        for bit 0 I need -,-,-,-,0,1,-,num_of_col,num_of_col+1,-,-,-,-,num_of_channel,num_of_channel+1,-,num_of_col,num_of_col+1
        for bit 1 I need _,_,_,0,1,2,num_of_col,num_of_col+1,num_of_col
        for bit n I need 
        """
        output_width = (floor((input_vars['Win']+2*stage['padding']-stage['Kw'])/stage['stride'])+1)
        bit_in_channel = bit_id % (output_width * output_width)
        first_row = floor((bit_in_channel * stage['stride']) / (input_vars['Win'] - stage['Kw'] + 2 * stage['padding'] + 1)) - stage[
            'padding']
        first_col = (bit_in_channel * stage['stride']) % (input_vars['Win'] - stage['Kw'] + 2 * stage['padding'] + 1) - \
                    stage[
                        'padding']
        for c in range(input_vars['Cin']):
            for i in range(first_row, first_row+stage['Kh']) :
                for j in range(first_col, first_col+stage['Kw']):
                    if i < 0 or i >= input_vars['Hin'] or j < 0 or j > j >= input_vars['Win']:
                        needed_data_bit_list.append(-1)
                    else:
                        needed_data_bit_list.append(j + i * input_vars['Win'] + c * input_vars['Win'] * input_vars['Hin'])
        for c in range(input_vars['Cin']):
            for i in range(stage['Kh']) :
                for j in range(stage['Kw']):
                    needed_kernel_bit_list.append(j + i * stage['Kw'] + c * stage['Kw'] * stage['Kh'])


    elif stage['type'] == 'max_pooling':
        """for bit 0 I need 0,1,num_of_col,num_of_col+1
           for bit 2 I need 4,5,num_of_col+4,num_of_col+5
           for bit n I need (Kw*n)+(floor((Kh*n)/num_of_col)*num_of_col*(Kw-1)"""
        first_bit_id = (stage['Kw'] * bit_id) + floor((stage['Kh']*bit_id)/input_vars['Win']) * \
            input_vars['Win']*(stage['Kw']-1)
        for i in range(stage['Kh']):
            for j in range(stage['Kw']):
                needed_data_bit_list.append(('data', first_bit_id+j+i*input_vars['Win']))
    else:
        # fully connect
        """
        It is dor each channel all the data and kernel
        """
        for j in range(stage['Kw']):
            needed_data_bit_list.append(j)
            needed_kernel_bit_list.append( j + bit_id * stage['Kw'])
    return needed_data_bit_list, needed_kernel_bit_list


def calculate_needed_bits_sub_convolution(input_vars, stage, bit_id, channels_list):
    needed_data_bit_list = []
    needed_kernel_bit_list = []

    output_width = (floor((input_vars['Win']+2*stage['padding']-stage['Kw'])/stage['stride'])+1)
    bit_in_channel = bit_id % (output_width * output_width)
    first_row = floor((bit_in_channel * stage['stride']) / (input_vars['Win'] - stage['Kw'] + 2 * stage['padding'] + 1)) - stage[
        'padding']
    first_col = (bit_in_channel * stage['stride']) % (input_vars['Win'] - stage['Kw'] + 2 * stage['padding'] + 1) - \
                stage[
                    'padding']
    for c in range(channels_list):
        for i in range(first_row, first_row+stage['Kh']):
            for j in range(first_col, first_col+stage['Kw']):
                if i < 0 or i >= input_vars['Hin'] or j < 0 or j > j >= input_vars['Win'] or c >= input_vars['Cin']:
                    needed_data_bit_list.append(-1)
                else:
                    needed_data_bit_list.append(j + i * input_vars['Win'] + c * input_vars['Win'] * input_vars['Hin'])
    for c in range(channels_list):
        for i in range(stage['Kh']):
            for j in range(stage['Kw']):
                if c >= input_vars['Cin']:
                    needed_kernel_bit_list.append(-1)
                else:
                    needed_kernel_bit_list.append(j + i * stage['Kw'] + c * stage['Kw'] * stage['Kh'])

    return needed_data_bit_list, needed_kernel_bit_list


def space_action(kernel_num_bit, sim_flags):
    # how much the input and the extra space needed for a specific action in a row
    space_needed = kernel_num_bit
    space_needed += 1  # first output
    if sim_flags['popcount_method'] == 'incrementor':
        init_num = ceil((pow(2, ceil(log(kernel_num_bit, 2))) - kernel_num_bit - 1) / 2)
        if init_num > 0:
            space_needed += ceil(log(init_num, 2) + 1)  # extra space for the intermediate outputs
        else:
            space_needed += 2
    elif sim_flags['popcount_method'] == 'tree_counter':
        space_needed += (ceil(kernel_num_bit / 3)) * 2  # for the first adding
    return space_needed

"""
For each Data/kernel in tha MMPU:
    ID - the identifier of the data bit - calculated as follows:
        column + row * max bit in a row + C * max bit in a channel
        example: 
        for bit in the third column of the 20th row in the 30th channel of 223*223*64(column*row*channel)
        3+20*223+30*(223*223) 
    Stage - The stage number 
    Row - the number of the row in the mat
    Mat - the number of Mat in the subarray
    Array - the number of subarray in the bank
    Bank - the number of bank in the chip
    Chip - the number of Chip in the rank
    Rank - the number of the rank 
    DataType - data/kernel/sub_summation
"""


"""
Bit_sequence for each action there is the data bits needed  
"""


class MmpuDataFrame:

    def __init__(self, structure, is_from_file):
        self.df_bits = pd.DataFrame(columns=['Bit_id', 'Stage', 'Row', 'Mat', 'Array', 'Bank', 'Chip', 'Rank', 'Data_type'])
        self.df_mat_center_of_gravity = pd.DataFrame(columns=['Mat', 'Array', 'Bank', 'Chip', 'Rank', 'Rows_id',
                                                              'Total_need'])
        self.df_rows_build = pd.DataFrame(columns=['Row_id', 'Kernel', 'Data_req_id', 'Output_bits', 'Num_sub_summation',
                                                   'Part_num'])
        self.df_data_req = pd.DataFrame(columns=['Data_id', 'Bit_id', 'Row_id'])
        self.df_mats_occupancy = pd.DataFrame(columns=['Mat', 'Array', 'Bank', 'Chip', 'Rank', 'Valid_bits', 'Taken'])
        self.df_status = pd.DataFrame(columns=['Stage', 'Total_move_latency', 'Total_action_latency', 'Total_move_power',
                                               'Total_action_power', 'Total_between_mats', 'Total_between_banks',
                                               'Total_between_chip', 'Total_between_ranks'])
        self.df_status = self.df_status.set_index('Stage')
        self.num_of_col = structure['rows_in_mat']
        self.num_of_mat = structure['mats_in_subarray']
        self.num_of_array = structure['subarrays_in_bank']
        self.num_of_bank = structure['banks_in_chips']
        self.num_of_chip = structure['chips_in_rank']
        self.num_of_rank = structure['num_of_rank']
        self.read_bandwidth = structure['read_bandwidth']
        self.data_req_id = 0
        if not is_from_file:
            for rank in range(self.num_of_rank):
                for chip in range(self.num_of_chip):
                    for bank in range(self.num_of_bank):
                        for array in range(self.num_of_array):
                            for mat in range(self.num_of_mat):
                                df_new_mat = pd.Series([mat, array, bank, chip, rank, 0, False],
                                                       index=self.df_mats_occupancy.columns)
                                self.df_mats_occupancy = self.df_mats_occupancy.append(df_new_mat, ignore_index=True)
        else:
            self.df_mats_occupancy = pd.read_csv('init_mmpu.csv')

    def init_mmpu(self, input_vars, first_stage):
        input_vars_new_dict = {
            'Kw': input_vars['Win'],
            'Kh': input_vars['Hin'],
            'Cout': input_vars['Cin']
        }
        self.insert_outside_data(0, input_vars_new_dict, 'data', 'first_empty')
        self.insert_outside_data(1, first_stage, 'kernel', 'first_empty')

    def kernel_needed_bits(self, input_vars, stage, bit_id):
        needed_kernel_bit_list = []
        if stage['type'] == 'convolution':
            """
            for bit 0 I need -,-,-,-,0,1,-,num_of_col,num_of_col+1,-,-,-,-,num_of_channel,num_of_channel+1,-,num_of_col,num_of_col+1
            for bit 1 I need _,_,_,0,1,2,num_of_col,num_of_col+1,num_of_col
            for bit n I need 
            """
            for c in range(input_vars['Cin']):
                for i in range(stage['Kh']):
                    for j in range(stage['Kw']):
                        needed_kernel_bit_list.append(j + i * stage['Kw'] + c * stage['Kw'] * stage['Kh'])

        elif stage['type'] == 'max_pooling':
                pass
        else:
            # fully connect
            """
            It is dor each channel all the data and kernel
            """
            for j in range(stage['Kw']):
                needed_kernel_bit_list.append(j + bit_id * stage['Kw'])

        return needed_kernel_bit_list

    def add_needed_data_bits(self, input_vars, stage, bit_id, row_id):
        needed_data_bit_list = []
        if stage['type'] == 'convolution':
            """
            for bit 0 I need -,-,-,-,0,1,-,num_of_col,num_of_col+1,-,-,-,-,num_of_channel,num_of_channel+1,-,num_of_col,num_of_col+1
            for bit 1 I need _,_,_,0,1,2,num_of_col,num_of_col+1,num_of_col
            for bit n I need 
            """
            output_width = (floor((input_vars['Win'] + 2 * stage['padding'] - stage['Kw']) / stage['stride']) + 1)
            bit_in_channel = bit_id % (output_width * output_width)
            first_row = floor(
                (bit_in_channel * stage['stride']) / (input_vars['Win'] - stage['Kw'] + 2 * stage['padding'] + 1)) - \
                        stage['padding']
            first_col = (bit_in_channel * stage['stride']) % (
                        input_vars['Win'] - stage['Kw'] + 2 * stage['padding'] + 1) - stage['padding']
            for c in range(input_vars['Cin']):
                for i in range(first_row, first_row + stage['Kh']):
                    for j in range(first_col, first_col + stage['Kw']):
                        if i < 0 or i >= input_vars['Hin'] or j < 0 or j > j >= input_vars['Win']:
                            df_new_data_req = pd.DataFrame([self.data_req_id, -1, row_id])
                        else:
                            df_new_data_req = pd.DataFrame([self.data_req_id, j + i * input_vars['Win'] +
                                                            c * input_vars['Win'] * input_vars['Hin'], row_id])
                        self.df_data_req = self.df_data_req.append(df_new_data_req)
                        needed_data_bit_list.append(self.data_req_id)
                        self.data_req_id += 1
        elif stage['type'] == 'max_pooling':
            """for bit 0 I need 0,1,num_of_col,num_of_col+1
               for bit 2 I need 4,5,num_of_col+4,num_of_col+5
               for bit n I need (Kw*n)+(floor((Kh*n)/num_of_col)*num_of_col*(Kw-1)"""
            first_bit_id = (stage['Kw'] * bit_id) + floor((stage['Kh'] * bit_id) / input_vars['Win']) * \
                           input_vars['Win'] * (stage['Kw'] - 1)
            for i in range(stage['Kh']):
                for j in range(stage['Kw']):
                    df_new_data_req = pd.DataFrame([self.data_req_id, first_bit_id + j + i * input_vars['Win'], row_id])
                    self.df_data_req = self.df_data_req.append(df_new_data_req)
                    needed_data_bit_list.append(self.data_req_id)
                    self.data_req_id += 1
        else:
            # fully connect
            """
            It is dor each channel all the data and kernel
            """
            for j in range(stage['Kw']):
                df_new_data_req = pd.DataFrame([self.data_req_id, j, row_id])
                self.df_data_req = self.df_data_req.append(df_new_data_req)
                needed_data_bit_list.append(self.data_req_id)
                self.data_req_id += 1

        return needed_data_bit_list

    def start_stage_reordering(self, stage, simulation_vars, space_time_var, input_vars, simulation_flags, stage_num,
                               structure_vars):
        sub_summation_flag = False
        # algorithm to find the least timing consuming transition
        self.df_rows_build = pd.DataFrame(columns=['Row_id', 'Kernel', 'Data_sequences', 'Output_bits',
                                                   'Num_sub_summation', 'Part_num'])
        self.df_mat_center_of_gravity = pd.DataFrame(columns=['Mat', 'Array', 'Bank', 'Chip', 'Rank', 'Rows_id'])
        max_num_bits_in_mat = self.num_of_col
        max_num_bits_for_gate = space_time_var['full_adder'][simulation_flags['orientation']]['space'] \
            if simulation_flags['orientation'] == 'tree_counter' \
            else space_time_var['half_adder'][simulation_flags['orientation']]['space']
        max_num_bits_for_kernel = stage['Kw'] * stage['Kh'] * input_vars['Cin']
        if stage['type'] == 'convolution':
            max_num_bits_for_action = space_action(max_num_bits_for_kernel, simulation_flags)
            output_width = (floor((input_vars['Win'] + 2 * stage['padding'] - stage['Kw']) / stage['stride']) + 1)
            if max_num_bits_in_mat > max_num_bits_for_gate + max_num_bits_for_kernel + max_num_bits_for_action:
                # no sub summation needed
                # at least one action can calculated inside a row
                max_num_bits_left_for_action = max_num_bits_in_mat - max_num_bits_for_gate - max_num_bits_for_kernel
                max_actions_in_row = max_num_bits_left_for_action / max_num_bits_for_action
                total_needed_bit = max_num_bits_for_gate + max_num_bits_for_kernel + \
                                   max_num_bits_for_action * max_actions_in_row
                in_current_row = 0
                previous_kernel = self.kernel_needed_bits(input_vars, stage, 0)
                row_id = 0
                rows_list = []
                data_sequences = []
                output_bits = []
                for c in stage['Cout']:
                    for i in range(output_width):
                        for j in range(output_width):
                            kernel_needed = self.kernel_needed_bits(input_vars, stage, j + i * output_width +
                                                                    c * output_width * output_width)
                            if kernel_needed != previous_kernel or in_current_row > max_actions_in_row:
                                # there can't be more than one kernel sequence in a row - for simplicity reason
                                df_new_row = pd.Series([row_id, previous_kernel, data_sequences, output_bits, 0, 0],
                                                       index=self.df_rows_build.columns)
                                rows_list.append(row_id)
                                row_id += 1
                                self.df_rows_build = self.df_rows_build.append(df_new_row, ignore_index=True)
                                previous_kernel = kernel_needed
                                output_bits = [j + i * output_width + c * output_width * output_width]
                                data_sequences = [needed_bits_id]
                                in_current_row = 0
                            else:
                                needed_bits_id = self.add_needed_data_bits(input_vars, stage, j + i * output_width
                                                                           + c * output_width * output_width, row_id)
                                data_sequences.append(needed_bits_id)
                                in_current_row += 1
                            if len(rows_list) == self.num_of_col:
                                mat, array, bank, chip, rank = self.calculate_center_of_mass(rows_list, stage_num)
                                df_new_mat_cog = pd.Series([mat, array, bank, chip, rank, rows_list, total_needed_bit],
                                                           index=self.df_mat_center_of_gravity.columns)
                                self.df_mat_center_of_gravity = self.df_mat_center_of_gravity.append(df_new_mat_cog)
                                rows_list = []
                # leftovers
                df_new_row = pd.Series([row_id, previous_kernel, data_sequences, output_bits, 0, 0],
                                       index=self.df_rows_build.columns)
                rows_list.append(row_id)
                self.df_rows_build = self.df_rows_build.append(df_new_row, ignore_index=True)
                if len(rows_list) != 0:
                    mat, array, bank, chip, rank = self.calculate_center_of_mass(rows_list, stage_num)
                    df_new_mat_cog = pd.Series([mat, array, bank, chip, rank, rows_list, total_needed_bit],
                                               index=self.df_mat_center_of_gravity.columns)
                    self.df_mat_center_of_gravity = self.df_mat_center_of_gravity.append(df_new_mat_cog,
                                                                                         ignore_index=True)

            else:
                # sub summation needed
                max_actions_in_row = 1
                sub_summation_flag = True
                action_space_of_channel = space_action(stage['Kw'] * stage['Kh'], simulation_flags)
                num_of_needed_bits = 0
                if action_space_of_channel + stage['Kw'] * stage['Kh'] + max_num_bits_for_gate < max_num_bits_in_mat:
                    num_of_rows = 2
                    for i in range(floor(log(input_vars['Cin'], 2))):
                        action_space = space_action(ceil(input_vars['Cin']/num_of_rows) * stage['Kw'] * stage['Kh'],
                                                    simulation_flags)
                        num_of_needed_bits = ceil(input_vars['Cin']/num_of_rows) * stage['Kw'] * stage['Kh'] + \
                                             action_space + max_num_bits_for_gate
                        if num_of_needed_bits < max_num_bits_in_mat:
                            break
                        num_of_rows *= 2

                    # now for each output bit there are 'num_of_rows' rows
                    num_kernel_channels_in_row = ceil(input_vars['Cin']/num_of_rows)
                    row_id = 0
                    rows_list = []
                    for c in stage['Cout']:
                        for i in range(output_width):
                            for j in range(output_width):
                                for splits in range(num_of_rows):
                                    channels_list = range(splits * num_kernel_channels_in_row,
                                                          (1 + splits) * num_kernel_channels_in_row)
                                    output_bit = j + i * output_width + c * output_width * output_width
                                    data_needed, kernel_needed = calculate_needed_bits_sub_convolution(input_vars, stage
                                                                                                       , output_bit,
                                                                                                       channels_list)
                                    df_new_row = pd.Series([row_id, kernel_needed, data_needed, output_bit,
                                                            len(kernel_needed), splits],
                                                           index=self.df_rows_build.columns)
                                    self.df_rows_build = self.df_rows_build.append(df_new_row, ignore_index=True)
                                    rows_list.append(row_id)
                                    row_id += 1

                                    if len(rows_list) == self.num_of_col:
                                        mat, array, bank, chip, rank = self.calculate_center_of_mass(rows_list,
                                                                                                     stage_num)
                                        df_new_mat_cog = pd.Series(
                                            [mat, array, bank, chip, rank, rows_list, num_of_needed_bits],
                                            index=self.df_mat_center_of_gravity.columns)
                                        self.df_mat_center_of_gravity = self.df_mat_center_of_gravity.append(
                                            df_new_mat_cog)
                                        rows_list = []
                    # leftovers
                    if len(rows_list) != 0:
                        mat, array, bank, chip, rank = self.calculate_center_of_mass(rows_list, stage_num)
                        df_new_mat_cog = pd.Series([mat, array, bank, chip, rank, rows_list, total_needed_bit],
                            index=self.df_mat_center_of_gravity.columns)
                        self.df_mat_center_of_gravity = self.df_mat_center_of_gravity.append(
                            df_new_mat_cog,
                            ignore_index=True)

                else:
                    # Currently assumes there will be at least one kernel computation in a row
                    print('convolution with big kernel - please implement')
                    exit(1)

        elif stage['type'] == 'max_pooling':
            pass
        else:
            # fully connect
            pass

        self.move_data(simulation_flags, stage_num, simulation_vars, structure_vars)

        return sub_summation_flag

    def activate_action(self, stage, simulation_vars, stage_num, max_actions_in_row):
        time_needed = 0
        power_needed = 0
        # count the number of actions made in parallel - the same number of rows neeeded - for power
        num_of_actions = len(self.df_rows_build.index)
        if stage['type'] == 'convolution' or stage['type'] == 'fully_connect':

            # need tha data and corresponding kernel bit and xnor outcome
            time_needed += kernel_num_bit * space_time_var['xnor'][sim_flags['orientation']]['time']
            power_needed += kernel_num_bit * power_gate(simulation_vars, space_time_var, sim_flags['orientation'], 'xnor')
            if sim_flags['popcount_method'] == 'incrementor':
                for p in range(1, kernel_num_bit + 1):
                    # added 1 below because for every new half adder we need to override space so one init
                    time_needed += (1 + space_time_var['half_adder'][sim_flags['orientation']]['time']) * math.ceil(
                        math.log(p, 2) + 1)
                    power_needed += math.ceil(math.log(p, 2) + 1) * \
                                    power_gate(simulation_vars, space_time_var, sim_flags['orientation'], 'half_adder')
            elif sim_flags['popcount_method'] == 'tree_counter':
                for p in range(math.ceil(math.log(kernel_num_bit, 2))):
                    # added 1 below beacuse for every new half adder we need to override space so one init
                    time_needed += space_time_var['full_adder'][sim_flags['orientation']]['time'] * (1 + p) * \
                                   math.floor(kernel_num_bit / (3 * math.pow(2, p)))
                    power_needed += (1 + p) * math.floor(kernel_num_bit / (3 * math.pow(2, p))) * \
                                    power_gate(simulation_vars, space_time_var, sim_flags['orientation'], 'full_adder')
        else:
            # max pooling
        pass


        self.df_status.at[stage_num, 'Total_action_latency'] += time_needed * max_actions_in_row
        self.df_status.at[stage_num, 'Total_action_power'] += power_needed * max_actions_in_row * num_of_actions

    def update_output_bits(self,stage_num):


    def find_first_empty_mat(self):
        df_valid_bits = self.df_mats_occupancy.loc[self.df_mats_occupancy['Valid_bits'] == 0]
        print(df_valid_bits.iloc[0])
        if not df_valid_bits.empty:
            return df_valid_bits.iloc[0].Mat, df_valid_bits.iloc[0].Array, df_valid_bits.iloc[0].Bank, \
                   df_valid_bits.iloc[0].Chip, df_valid_bits.iloc[0].Rank
        return None, None, None, None, None
    """
    Data's center of mass:
    for all the data and kernel needed to be inserted into the mat we calculate the following:
    we find the mat/subarray/bank/chip/rank that most of the data is in
    this way we can calculate how much it will cost to move the mat to someplace else:
    the same mat, mat in the same subarray, mat in the same bank ...
    """
    def calculate_center_of_mass(self, rows_list, stage_num):
        data_bit_set = set()
        for row in self.df_rows_build.loc[rows_list]:
            for bit_sequence in row.Data_sequences:
                for bit_id in bit_sequence:
                    if bit_id == -1:  # this is a zero in the array. no need to bring data
                        continue
                    data_bit_set.add(bit_id)
        data_bit_list = list(data_bit_set)
        mean_rows = self.df_bits.loc[self.df_bits.loc["Bit_id"].isin(data_bit_list) &
                                     self.df_bits.loc["Stage"] == stage_num - 1].mean()
        mat = mean_rows.Mat
        array = mean_rows.Array
        bank = mean_rows.Bank
        chip = mean_rows.Chip
        rank = mean_rows.rank
        return mat, array, bank, chip, rank

    def find_best_available_mat(self, df_mat_center_of_mass, simulation_vars):
        bits_needed = df_mat_center_of_mass.Total_needed
        chip_mass = df_mat_center_of_mass.Chip
        bank_mass = df_mat_center_of_mass.Bank
        rank_mass = df_mat_center_of_mass.Rank
        mat = None
        array = None
        bank = None
        chip = None
        rank = None
        if simulation_vars['latency']['write_between_mats'] < simulation_vars['latency']['write_between_banks']:
            df_mat_in_same_bank = self.df_mats_occupancy.loc[((self.df_mats_occupancy['Bank'] == bank_mass &
                                                               self.df_mats_occupancy['Chip'] == chip_mass) &
                                                              (self.df_mats_occupancy['Rank'] == rank_mass &
                                                               self.df_mats_occupancy['Taken'] is False)) &
                                                             self.df_mats_occupancy['Valid_bits'] < self.num_of_col -
                                                             bits_needed]
            if not df_mat_in_same_bank.empty:
                return df_mat_in_same_bank[0].Mat, df_mat_in_same_bank[0].Array, \
                       df_mat_in_same_bank[0].Bank, df_mat_in_same_bank[0].Chip, df_mat_in_same_bank[0].Rank

            df_mat_in_same_chip = self.df_mats_occupancy.loc[((self.df_mats_occupancy['Bank'] != bank_mass &
                                                               self.df_mats_occupancy['Chip'] == chip_mass)) &
                                                             (self.df_mats_occupancy['Rank'] == rank_mass &
                                                              self.df_mats_occupancy['Taken'] is False) &
                                                             self.df_mats_occupancy['Valid_bits'] < self.num_of_col -
                                                             bits_needed]
            if not df_mat_in_same_chip.empty:
                return df_mat_in_same_chip[0].Mat, df_mat_in_same_chip[0].Array, \
                       df_mat_in_same_chip[0].Bank, df_mat_in_same_chip[0].Chip, df_mat_in_same_chip[0].Rank
        else:
            df_mat_in_same_chip = self.df_mats_occupancy.loc[((self.df_mats_occupancy['Bank'] != bank_mass &
                                                               self.df_mats_occupancy['Chip'] == chip_mass)) &
                                                             (self.df_mats_occupancy['Rank'] == rank_mass &
                                                              self.df_mats_occupancy['Taken'] is False) &
                                                             self.df_mats_occupancy['Valid_bits'] < self.num_of_col -
                                                             bits_needed]
            if not df_mat_in_same_chip.empty:
                return df_mat_in_same_chip[0].Mat, df_mat_in_same_chip[0].Array, \
                       df_mat_in_same_chip[0].Bank, df_mat_in_same_chip[0].Chip, df_mat_in_same_chip[0].Rank

            df_mat_in_same_bank = self.df_mats_occupancy.loc[((self.df_mats_occupancy['Bank'] == bank_mass &
                                                               self.df_mats_occupancy['Chip'] == chip_mass) &
                                                              (self.df_mats_occupancy['Rank'] == rank_mass &
                                                               self.df_mats_occupancy['Taken'] is False)) &
                                                             self.df_mats_occupancy['Valid_bits'] < self.num_of_col -
                                                             bits_needed]
            if not df_mat_in_same_bank.empty:
                return df_mat_in_same_bank[0].Mat, df_mat_in_same_bank[0].Array, \
                       df_mat_in_same_bank[0].Bank, df_mat_in_same_bank[0].Chip, df_mat_in_same_bank[0].Rank

        if mat is None:
            # couldn't find mat in the same bank or chip search for a mat in a close chip
            df_mat_in_same_rank = self.df_mats_occupancy.loc[(self.df_mats_occupancy['Chip'] != chip_mass &
                                                              self.df_mats_occupancy['Rank'] == rank_mass) &
                                                             (self.df_mats_occupancy['Taken'] is False &
                                                              self.df_mats_occupancy['Valid_bits'] < self.num_of_col -
                                                              bits_needed)]
            if not df_mat_in_same_rank.empty:
                return df_mat_in_same_rank[0].Mat, df_mat_in_same_rank[0].Array, \
                       df_mat_in_same_rank[0].Bank, df_mat_in_same_rank[0].Chip, df_mat_in_same_rank[0].Rank

        if mat is None:
            # couldn't find mat in the rank search for a mat in all the rank
            df_mat_all_the_ranks = self.df_mats_occupancy.loc[self.df_mats_occupancy['Rank'] != rank_mass &
                                                              (self.df_mats_occupancy['Taken'] is False &
                                                               self.df_mats_occupancy['Valid_bits'] < self.num_of_col -
                                                               bits_needed)]
            if not df_mat_all_the_ranks.empty:
                return df_mat_all_the_ranks[0].Mat, df_mat_all_the_ranks[0].Array, \
                       df_mat_all_the_ranks[0].Bank, df_mat_all_the_ranks[0].Chip, df_mat_all_the_ranks[0].Rank

        return mat, array, bank, chip, rank

    def move_mat(self, src_mat_row, bank, chip, rank, simulation_vars, simulation_flags, stage_num, df_bank_level_read):
        cycles_needed = 0
        power_needed = 0
        df_mat_data_req = self.df_data_req.loc[self.df_data_req['Row_id'].isin(src_mat_row.Rows_id)]
        bits_count = len(df_mat_data_req)
        df_this_bank_level_read = df_bank_level_read.loc[((df_bank_level_read['Bank'] == bank) &
                                                         (df_bank_level_read['Chip'] == chip)) &
                                                         (df_bank_level_read['Rank'] == rank)]
        # calculating all th write needed in the mat
        if simulation_flags['multiple_writes']:
            # groups all the bits from the same write group:
            #       1. they are in the same row in the destination mat
            #       2. they can be read in the same cycle
            df_data_seq = df_mat_data_req.groupby(['Read_num', 'Row_id'])['Bit_id'].apply(list).reset_index(
                name='Id_list')
        else:
            # groups all the bits from the same write group:
            #       1. they are in consecutive places in the destination mat
            #       2. they are in the same row in the destination mat
            #       3. they can be read in the same cycle
            df_data_seq = df_mat_data_req.groupby(['Write_group'])['Bit_id'].apply(list).reset_index(name='Id_list')

        num_write_to_mat = len(df_data_seq)
        # the move process:
        #   1. read the bits from the mat into the bank IO (happen once per read snippet)
        #   2. move the bits to the needed bank IO (happen once per read snippet)
        #   3. write into the rows of the mats inside the bank (Happen as much as needed )

        df_write_in_same_bank = df_data_seq.loc[(df_data_seq['Bank'] == bank) & ((df_data_seq['Chip'] == chip) &
                                                                                 (df_data_seq['Rank'] == rank))]
        in_bank_movements = len(df_write_in_same_bank.index)

        df_write_in_same_chip = df_data_seq.loc[(df_data_seq['Bank'] != bank) & ((df_data_seq['Chip'] == chip) &
                                                                                 (df_data_seq['Rank'] == rank))]
        df_write_in_same_chip['Moved'] = (df_write_in_same_chip['Read_num'].isin(list(df_this_bank_level_read.Read_set)))
        need_to_move_in_chip = (~df_write_in_same_chip['Moved']).values.sum()

        df_write_in_same_rank = df_data_seq.loc[(df_data_seq['Rank'] == rank) & (df_data_seq['Chip'] != chip)]
        df_write_in_same_rank['Moved'] = (df_write_in_same_rank['Read_num'].isin(list(df_this_bank_level_read.Read_set)))
        need_to_move_in_rank = (~df_write_in_same_rank['Moved']).values.sum()

        df_write_between_ranks = df_data_seq.loc[(df_data_seq['Rank'] != rank)]
        df_write_between_ranks['Moved'] = (df_write_between_ranks['Read_num'].isin(list(df_this_bank_level_read.Read_set)))
        need_to_move_between_ranks = (~df_write_between_ranks['Moved']).values.sum()

        read_num_list = df_data_seq.Read_num.unique()
        df_bank_level_read.loc[((df_bank_level_read['Bank'] == bank) &
                                (df_bank_level_read['Chip'] == chip)) &
                               (df_bank_level_read['Rank'] == rank), 'Read_set'] = \
            df_this_bank_level_read.Read_set.update(read_num_list)
        num_write_to_mat = len(df_data_seq.index) - in_bank_movements
        self.df_status.at[stage_num, 'Total_between_mats'] += in_bank_movements
        self.df_status.at[stage_num, 'Total_between_banks'] += need_to_move_in_chip
        self.df_status.at[stage_num, 'Total_between_chip'] += need_to_move_in_rank
        self.df_status.at[stage_num, 'Total_between_ranks'] += need_to_move_between_ranks

        cycles_needed += in_bank_movements * simulation_vars['latency']['write_between_mats'] + \
                         need_to_move_in_chip * simulation_vars['latency']['move_between_banks'] + \
                         need_to_move_in_rank * simulation_vars['latency']['move_between_chip'] + \
                         need_to_move_between_ranks * simulation_vars['latency']['move_between_ranks'] + \
                         num_write_to_mat * simulation_vars['latency']['write_from_bank_io']

        power_needed += in_bank_movements * simulation_vars['power']['write_between_mats'] + \
                        need_to_move_in_chip * simulation_vars['power']['move_between_banks'] + \
                        need_to_move_in_rank * simulation_vars['power']['move_between_chip'] + \
                        need_to_move_between_ranks * simulation_vars['power']['move_between_ranks'] + \
                        num_write_to_mat * simulation_vars['power']['write_from_bank_io'] + \
                        bits_count * simulation_vars['power']['write_per_bit']

        self.df_status.at[stage_num, 'Total_move_latency'] += cycles_needed
        self.df_status.at[stage_num, 'Total_move_power'] += power_needed

    def move_data(self,simulation_flags, stage_num, simulation_vars, structure_vars):
        cycles_needed = 0
        power_needed = 0
        # for each mat, find the first possible mat
        df_bits_temp = self.df_bits.iloc[(self.df_bits['Stage'] == stage_num-1) &
                                                  self.df_bits['Type'].str.contains('Data')]
        df_bits_temp.sort_values(by=['Row', 'Mat', 'Array', 'Bank', 'Chip', 'Rank'], inplace=True)
        df_bits_temp['Row_num'] = ((((df_bits_temp['Mat'] != df_bits_temp['Mat'].shift()) |
                                     (df_bits_temp['Row'] != df_bits_temp['Row'].shift())) |
                                    ((df_bits_temp['Bank'] != df_bits_temp['Bank'].shift()) |
                                     (df_bits_temp['Chip'] != df_bits_temp['Chip'].shift()))) |
                                   (df_bits_temp['Rank'] != df_bits_temp['Rank'].shift())).cumsum()
        df_bits_temp['Bit_in_row'] = df_bits_temp.groupby('Row_num').cumcount
        if simulation_flags['Read_style'] == 'row':
            df_bits_temp['Read_num'] = ((df_bits_temp['Bit_in_row'] == structure_vars['read_bandwidth'] - 1) |
                                        df_bits_temp['Row_num'] != df_bits_temp['Row_num'].shift()).cumsum()
            df_bits_temp.drop(columns=['Row_num', 'Bit_in_row'])
        elif simulation_flags['Read_style'] == 'col': # reading in columns could lead to higher read bandwidth
            df_bits_temp['Mat_num'] = (((df_bits_temp['Mat'] != df_bits_temp['Mat'].shift()) |
                                        ((df_bits_temp['Bank'] != df_bits_temp['Bank'].shift()) |
                                         (df_bits_temp['Chip'] != df_bits_temp['Chip'].shift()))) |
                                       (df_bits_temp['Rank'] != df_bits_temp['Rank'].shift())).cumsum()
            df_bits_temp['Bit_in_col'] = df_bits_temp.groupby(['Bit_in_row', 'Mat', 'Array', 'Bank', 'Chip', 'Rank']).cumcount
            df_bits_temp.sort_values(by=['Bit_in_row', 'Mat', 'Array', 'Bank', 'Chip', 'Rank'], inplace=True)
            df_bits_temp['Read_num'] = ((df_bits_temp['Bit_in_col'] == structure_vars['read_bandwidth'] - 1) |
                                        df_bits_temp['Mat_num'] != df_bits_temp['Mat_num'].shift()).cumsum()
            df_bits_temp.drop(columns=['Row_num', 'Bit_in_row','Bit_in_col','Mat_num'])
        else:
            print('-E- Wrong Read_style choice')
            exit(1)

        self.df_data_req = self.df_data_req.merge(df_bits_temp, how='left', on='Bit_id')
        self.df_data_req['Write_group'] = ((self.df_data_req['Bit_id'] != self.df_data_req['Bit_id'].shift(1) + 1) |
                                        (self.df_data_req['Row_id'] != self.df_data_req['Row_id'].shift(1)) |
                                        (self.df_data_req['Read_num'] != self.df_data_req['Read_num'].shift(1))).cumsum()

        # read of all the possible read snippets into bank IO
        df_read_needed = self.df_data_req.groupby('Read_num')['Bit_id'].apply(list).reset_index(name='Id_list')
        read_needed_num = len(df_read_needed.index)
        cycles_needed += simulation_vars['latency_read_to_bank_io'] * read_needed_num
        power_needed += simulation_vars['power_read_to_bank_io'] * read_needed_num

        df_bank_level_read = pd.DataFrame(columns=['Bank', 'Chip', 'Rank', 'Read_set'])
        for index, mat_row in self.df_mat_center_of_gravity.iterrows():
            if simulation_flags['find_best_mat']:
                mat, array, bank, chip, rank = self.find_best_available_mat(mat_row, simulation_vars)
            else:
                mat, array, bank, chip, rank = self.find_first_empty_mat()
            if mat is not None:
                self.df_mats_occupancy.loc[((self.df_mats_occupancy['Mat'] == mat &
                                             self.df_mats_occupancy['Array'] == array) &
                                            (self.df_mats_occupancy['Bank'] == bank &
                                             self.df_mats_occupancy['Chip'] == chip)) &
                                           (self.df_mats_occupancy['Rank'] == rank), ['Taken']] = True
            else:
                print("-E- no place to fit the next stage - need to add stage in parts")
                exit(1)
            if df_bank_level_read.loc[(df_bank_level_read.Bank == bank) &
                                      ((df_bank_level_read.Chip == chip) & (df_bank_level_read.Rank == rank))].empty:
                df_new_bank_set = pd.Series([bank, chip, rank, set()], index=df_bank_level_read.columns)
                df_bank_level_read = df_bank_level_read.append(df_new_bank_set, ignore_index=True)
            # move the data to its designated mat
            self.move_mat(mat_row, bank, chip, rank, simulation_vars, simulation_flags, stage_num, df_bank_level_read)

    def insert_outside_data(self, stage_num, data_parameters, data_type, way_of_placement):
        if way_of_placement == 'first_empty':
            mat_inserted, array_inserted, bank_inserted, chip_inserted, rank_inserted = self.find_first_empty_mat()
            if mat_inserted is None:
                print('-E- there is no empty mats left')
                exit(1)
            bit_inserted = 0
            row_inserted = 0

            for c in range(data_parameters['Cout']):
                for j in range(data_parameters['Kh']):
                    for i in range(data_parameters['Kw']):
                        df_new_bit = pd.Series([i + j * data_parameters['Kw'] + c * data_parameters['Kw'] * data_parameters['Kh'],
                                                   stage_num, row_inserted, mat_inserted, array_inserted,
                                                   bank_inserted, chip_inserted, rank_inserted, data_type],
                                                  index=self.df_bits.columns)
                        self.df_bits = self.df_bits.append(df_new_bit, ignore_index=True)
                        bit_inserted += 1
                        if bit_inserted >= self.num_of_col:
                            row_inserted += 1
                            bit_inserted = 0
                            if row_inserted == 1:
                                self.df_mats_occupancy.loc[((((self.df_mats_occupancy['Mat'] == mat_inserted) &
                                                              (self.df_mats_occupancy['Array'] == array_inserted)) &
                                                            ((self.df_mats_occupancy['Bank'] == bank_inserted) &
                                                             (self.df_mats_occupancy['Chip'] == chip_inserted))) &
                                                           (self.df_mats_occupancy['Rank'] == rank_inserted)),
                                                           'Valid_bits'] = self.num_of_col
                            if row_inserted >= self.num_of_col:
                                mat_inserted, array_inserted, bank_inserted, chip_inserted, rank_inserted = \
                                    self.find_first_empty_mat()
                                if mat_inserted is None:
                                    print('-E- there is no empty mats left')
                                    exit(1)
                                row_inserted = 0

            # if wrote inside a mat but only on the first line. The mat is not empty now.
            if bit_inserted > 0 and row_inserted == 0:
                self.df_mats_occupancy.loc[((((self.df_mats_occupancy['Mat'] == mat_inserted) &
                                              (self.df_mats_occupancy['Array'] == array_inserted)) &
                                            ((self.df_mats_occupancy['Bank'] == bank_inserted) &
                                             (self.df_mats_occupancy['Chip'] == chip_inserted))) &
                                           (self.df_mats_occupancy['Rank'] == rank_inserted)),
                                           ['Valid_bits']] = bit_inserted



    def print_bits(self):
        print(self.df_bits)

    def print_occupance(self):
        print(self.df_mats_occupancy)

    def print_same_row_bits(self):
        df_same_row_bits = self.df_bits.groupby(['Mat', 'Array', 'Bank', 'Chip', 'Rank'])['Bit_id'].apply(list).reset_index(name='Id_list')
        print(df_same_row_bits)
    def print_occupancy_not_zero(self):
        print(self.df_mats_occupancy.loc[self.df_mats_occupancy['Valid_bits'] != 0])

    def save_situation(self,stage_num):
        self.df_mats_occupancy.to_csv('mats_occupancy.csv', index=False)
        self.df_bits.to_csv('bits.csv', index=False)
        self.df_mats_occupancy.to_csv('mats_occupancy_'+stage_num+'.csv', index=False)
        self.df_bits.to_csv('bits_'+stage_num+'.csv', index=False)

    def load_situation(self):
        self.df_mats_occupancy = pd.read_csv('mats_occupancy.csv')
        self.df_bits = pd.read_csv('bits.csv')

    def present_mmpu(self, stage_num):
        # This function present the mmpu
        # first is to transfer the DataFrames
        return