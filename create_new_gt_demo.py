import os
import os.path as osp

# Change this path to the users own dataset path
desktop_path = os.path.expanduser("~\Desktop")
seq_path = os.path.join(desktop_path, "dataset", 'MOT')

seqlist_path = "sequence_groups"

def main():
    # Set the name of sequences for tracking
    seqlist_name = "seq_list6.txt"
    seq_file_path = os.path.join(seqlist_path, seqlist_name)
    lines = [line.rstrip('\n').split(' ') for line in open(seq_file_path) if len(line) > 1]
    seq_names = []
    for line in lines:
        seq_names.append(line[0])

    for seq_name in seq_names:
        gt_path = osp.join(seq_path, seq_name, 'gt', 'gt.txt')

        assert osp.exists(gt_path), "No ground-truth exists for {}".format(seq_name)

        with open(gt_path, 'r') as gt_file:
            lines = gt_file.readlines()
            for line in lines:
                contents = line.rstrip('\n').split(',')
                if int(contents[0]) == 1:

if __name__=="__main__":
    main()