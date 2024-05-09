import os
import csv

# Put machine_id just in case but it should be useful for now
fields = ['filename', 'normal', 'machine_id']

filename = "./data/test.csv"

dir_path = "./data/test"
dir = os.fsencode(dir_path)

my_rows = []
for file in os.listdir(dir):
    fn = os.fsdecode(file)
    split_fn = fn.split("_")
    row = {}

    row['filename'] = fn

    if split_fn[0] == 'normal':
        row['normal'] = 1
    else:
        row['normal'] = 0
    
    row['machine_id'] = split_fn[2]

    my_rows.append(row)


with open(filename, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(my_rows)