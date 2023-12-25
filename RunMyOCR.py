from test import runOCR
import pickle

def main():
    path_to_test = './HW4/images/test.bmp'
    path_to_Ytrue = './HW4/test_gt_py3.pkl'
    
    runOCR(path_to_test, path_to_Ytrue)
    
if __name__ == '__main__':
    main()