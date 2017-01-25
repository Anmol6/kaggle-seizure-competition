import sys, getopt
import os.path
import pdb

from scipy.io import loadmat
import numpy as np

def main(argv):
    inputf = ''
    outputdir = ''
    try:
        opts, args = getopt.getopt(argv, "i:o:", ["ifile=","odir="])
    except getopt.GetoptError:
        print( 'test.py -i <inputfile> -o <outputdir>' )
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputf = arg
        elif opt in ("-o", "--odir"):
            outputdir = arg
    
    if(not inputf or not outputdir):
        print( 'test.py -i <inputfile> -o <outputdir>' )
        sys.exit(2)

    if(os.path.isdir(inputf)):
        i = 0
        success = 0
        for filename in os.listdir(inputf):
            base, _ = os.path.splitext(os.path.basename(filename))
            if( filename.endswith('.mat') and (base + '.npy' not in os.listdir(outputdir)) ):
                i+=1
                print(str(i) + ' Converting: ' + filename)
                if(save_conversion( os.path.join(inputf, filename), os.path.join(outputdir, base + '.npy'))):
                    success+=1
        print('Total successful: ' + str(success) + '/' + str(i))
    else:
        base, _ = os.path.splitext(os.path.basename(inputf))
        print('Converting: ' + inputf)
        save_conversion(inputf, os.path.join(outputdir, base + '.npy'))

def save_conversion(mat_file, dest_file):
    try:
        np_data = loadmat(mat_file)
        np.save(dest_file, np_data['dataStruct'][0][0])
        return True
    except:
        print('Could not convert: ' + mat_file)
        return False
    # pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv[1:])
