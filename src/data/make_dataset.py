import yaml
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    "Load your data set from given file path"
    return pd.read_csv(file_path)

def split_data(data, test_size,seed):
    "split the data into trin and test sets"
    train,test=train_test_split(data,test_size=test_size,random_state=seed)
    return train,test

def save_data(train,test,output_path):
    "if path file not exist create it"
    Path(output_path).mkdir(parents=True,exist_ok=True)
    "saving the data to the give output path"
    train.to_csv(output_path +"/train.csv",index=False)
    test.to_csv(output_path+"/test.csv",index=False)

def main():
    curr_dir=Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    params_file=home_dir.as_posix()+'/params.yaml'
    params=yaml.safe_load(open(params_file))['make_dataset']

    
    input_path=home_dir.as_posix()+sys.argv[1]

    output_path=home_dir.as_posix()+sys.argv[2]

    data=load_data(input_path)
    train,test=split_data(data,params['test_size'],params['seed'])
    save_data(train,test,output_path)

    

if __name__ == '__main__':
    main()
