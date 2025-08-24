from pathlib import Path
import pandas as pd
import sys
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
# "Class"

def train_model(train_features,target,n_estimators,max_depth,seed):
    model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=seed)
    model.fit(train_features,target)
    return model

def save_model(model,output_path):
    joblib.dump(model,output_path+"/model.joblib")

def main():
    curr_dir=Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    params_file=home_dir.as_posix()+"/params.yaml"
    params=yaml.safe_load(open(params_file))['train_model']

    data_path=home_dir.as_posix()+sys.argv[1]

    output_path=home_dir.as_posix()+sys.argv[2]
    Path(output_path).mkdir(parents=True,exist_ok=True)

    Target=sys.argv[3]
    data=pd.read_csv(data_path)
    x=data.drop(Target,axis=1)
    y=data[Target]

    trained_model=train_model(x,y,params['n_estimators'],params['max_depth'],params['seed'])
    save_model(trained_model,output_path)

if __name__=="__main__":
    main()