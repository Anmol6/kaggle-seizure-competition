import csv
import sys, getopt
import os

class SafeDataFilter: 
    def __init__(self, csv_path = os.path.join(os.path.dirname(__file__), 'train_and_test_data_labels_safe.csv')):
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
    
    def get_label(self, filen):
        namef, _ = os.path.splitext(os.path.basename(filen))
        filenames = zip(*self.safelist)[0]
        try:
            i = filenames.index(namef)
            return self.safelist[i][1]
        except:
            return None

def main(argv):
    inputf = ''
    outputf = ''
    try:
        opts, args = getopt.getopt(argv, "i:o:", ["idir=", "odir"])
    except getopt.GetoptError:
        print( 'test.py -i <inputdir> -o <outputdir>' )
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-i", "--idir"):
            inputf = arg
    if opt in ("-o", "--odir"):
            outputf = arg
    
    if(not inputf):
        print( 'test.py -i <inputdir>' )
        sys.exit(2)

    # print out safe files
    dataf = SafeDataFilter()
    if(os.path.isdir(inputf)):
        listf = dataf.filter_safe([os.path.join(inputf, f) for f in os.listdir(inputf)])
        for filename in listf:
            print(filename)
            if(outputf):
                print(os.path.join(outputf, os.path.basename(filename[0])))
                os.rename(filename[0], os.path.join(outputf, os.path.basename(filename[0])))

if __name__ == "__main__":
    main(sys.argv[1:])
