import csv
import sys, getopt
import os

class SafeDataFilter: 
    def __init__(self, csv_path = 'train_and_test_data_labels_safe.csv'):
        self.safelist = []
        with open(csv_path) as csvf:
            reader = csv.DictReader(csvf)
            for row in reader:
                if(row['safe'] == '1'):
                    row_name, _ = os.path.splitext(os.path.basename(row['image']))
                    self.safelist.append( (row_name, row['class']) )

    def filter_safe(self, listf=None):
        '''
        Takes a list of file names/paths and returns the ones that are safe training data 
        with their label (according to train_and_test_data_labels_safe.csv).
        [(filename, class)]
        If listf is not specified, returns all safe files instead.
        '''
        safelistf = []
        if listf is None:
            return self.safelist
        else:
            filenames = zip(*self.safelist)[0]
            for filen in listf:
                # Extract just the name
                namef, _ = os.path.splitext(os.path.basename(filen))
                try:
                    i = filenames.index(namef)
                    safelistf.append( (filen, self.safelist[i][1]) )
                except:
                    pass
        return safelistf

def main(argv):
    inputf = ''
    try:
        opts, args = getopt.getopt(argv, "i:", ["ifile="])
    except getopt.GetoptError:
        print( 'test.py -i <inputfile>' )
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputf = arg
    
    if(not inputf):
        print( 'test.py -i <inputfile>' )
        sys.exit(2)

    # print out safe files
    dataf = SafeDataFilter()
    if(os.path.isdir(inputf)):
        listf = dataf.filter_safe(os.listdir(inputf))
        for filename in listf:
            print(filename)

if __name__ == "__main__":
    main(sys.argv[1:])
