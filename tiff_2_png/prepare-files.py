import argparse
import sys
import os 

def sequence_files(path_to_foder, posfix ="sq", sequence_start = 1):
    num_files = len(os.listdir(path_to_foder))
    digits = len(str(num_files))
    for fname in os.listdir(path_to_foder):
        sq_digits = len(str(sequence_start))
        name, extension = os.path.splitext(fname)
        file_posfix = "_"+posfix
        file_posfix = file_posfix.ljust(digits - sq_digits + len(file_posfix), '0')+str(sequence_start)
        name = name+file_posfix
        os.rename(path_to_foder+fname,path_to_foder+name+extension)
        sequence_start = sequence_start + 1
    print("sequence completed")   



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renamed",
                        help="Rename all files in a directory")
    parser.add_argument("-s","--source",
                        help="new name for files in directory")
    parser.add_argument("--sequence",
                        help="new name for files in directory",action='store_true')
    parser.add_argument("--sq_prefix",
                        help="prefix for new named sequence")
    parser.add_argument("--sq_start",
                        help="prefix for new named sequence")
    
    args = parser.parse_args()
    
    if args.sequence and os.path.exists(args.source):
        if args.sq_prefix:
            sequence_files(args.source,args.sq_prefix)
        elif args.sq_prefix and args.sq_start:
            sequence_files(args.source,args.sq_prefix,args.sq_start)
        else:
            sequence_files(args.source)
         

if __name__ == '__main__':
    main()