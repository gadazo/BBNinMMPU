
import csv
import sim_helper


structure_vars = {
    'mats_in_subarray': 16,
    'subarrays_in_bank': 64,
    'banks_in_chips': 8,
    'chips_in_rank': 8,
    'num_of_rank': 2,
    'rows_in_mat': 1024,
    'read_bandwidth': 16
}

input_vars = {
    'Win': 224,
    'Hin': 224,
    'Cin': 3,
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
"""
with open('init_mmpu.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Mat', 'Array', 'Bank', 'Chip', 'Rank', 'Valid_bits', 'Taken'])
    for rank in range(structure_vars['num_of_rank']):
        for chip in range(structure_vars['chips_in_rank']):
            for bank in range(structure_vars['banks_in_chips']):
                for array in range(structure_vars['subarrays_in_bank']):
                    for mat in range(structure_vars['mats_in_subarray']):
                        writer.writerow([mat, array, bank, chip, rank, 0, False])

print("wrote CSV")
"""

new_mmpu = sim_helper.MmpuDataFrame(structure_vars, True)
new_mmpu.print_occupancy_not_zero()
print("done init")

#new_mmpu.init_mmpu(input_vars, VGG16_program[0]) # did it before can be uploaded
new_mmpu.load_situation()
new_mmpu.print_occupancy_not_zero()
new_mmpu.print_bits()
new_mmpu.print_same_row_bits()
#for i in range(start_stage)
#print("done inserting bits")
#new_mmpu.save_situation()
