import os



data_dir = 'data/DPR_dataset'
img_names = set(list(os.listdir(data_dir)))

file_list = ['data/test.lst', 'data/train.lst', 'data/val.lst']

for file in file_list:
    print(file)
    with open(file, 'r') as f:
        for line in f:
            img_name = line.split()[0]
            if img_name not in img_names:
                print(img_name) # missing imgHQ29256

with open('data/train_n.lst', 'w') as f:
    with open('data/train.lst', 'r') as f2:
        for line in f2:
            img_name = line.split()[0]
            if img_name in img_names:
                f.write(line)

old_train_line_num = len(open('data/train.lst', 'r').readlines())
new_train_line_num = len(open('data/train_n.lst', 'r').readlines())
print(old_train_line_num, new_train_line_num)
                

