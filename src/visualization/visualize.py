from pathlib import Path
import joblib
import sys
import yaml

import pandas as pd
from sklearn import metrics,tree
from dvclive import Live
import matplotlib.pyplot as plt

def evaluate_model(model,test_features,test_target,split,live,output_path):
    predictions_by_class=model.predict_proba(test_features)
    predictions=predictions_by_class[:,1]

    avg_precision=metrics.average_precision_score(test_target,predictions)
    roc_auc=metrics.roc_auc_score(test_target,predictions)
   
    

    if not live.summary:
        live.summary={"avg_precision":{},"roc_auc":{}}
        live.summary["avg_precision"][split]=avg_precision
        live.summary["roc_auc"][split]=roc_auc
    

    live.log_sklearn_plot("precision_recall",test_target,predictions,name=f"prc/{split}")
    live.log_sklearn_plot("roc",test_target,predictions,name=f"roc/{split}")
    live.log_sklearn_plot("confusion_matrix",test_target,predictions,name=f"cm/{split}")
    






def save_importance_plot(live,model,feature_names):
    fig,ax=plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2,top=0.9)
    ax.set_ylabel("Mean Decrease in Impurity",fontsize=10)
    ax.set_title("Feature Importance",fontsize=10)

    importance=model.feature_importances_
    forest_importances=pd.Series(importance,index=feature_names).nlargest(n=10)
    forest_importances.plot.bar(ax=ax)
    live.log_image("importance.png",fig)


def main():
    curr_dir=Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    
    model_path=home_dir.as_posix()+sys.argv[1]

    model=joblib.load(model_path)

    data_path=home_dir.as_posix()+sys.argv[2]
    output_path=home_dir.as_posix()+sys.argv[3]
    Path(output_path).mkdir(parents=True,exist_ok=True)

    Target=sys.argv[4]
    df_train=pd.read_csv(data_path+'/train.csv')
    df_test=pd.read_csv(data_path+'/test.csv')

    x_train=df_train.drop(Target,axis=1)
    y_train=df_train[Target]
    x_test=df_test.drop(Target,axis=1)
    y_test=df_test[Target]

    with Live(output_path,dvcyaml=False ) as live:
        evaluate_model(model,x_train,y_train,"train",live,output_path)
        evaluate_model(model,x_test,y_test,"test",live,output_path)
        save_importance_plot(live,model,x_train.columns.to_list())

if __name__=="__main__":
    main()
         
