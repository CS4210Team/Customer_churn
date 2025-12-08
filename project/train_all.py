# project/train_all.py
from project.model_utils.logreg.train import train_logreg
from project.model_utils.tree.train import train_tree
from project.model_utils.knn.train import train_knn
from project.model_utils.svm.train import train_svm


def train_all_models():
    print("===== TRAINING ALL MODELS =====\n")
    
    train_logreg()
    print("\n-------------------------------\n")
    
    train_tree()
    print("\n-------------------------------\n")
    
    train_knn()
    print("\n-------------------------------\n")
    
    train_svm()
    print("\n===== ALL MODELS TRAINED =====")

if __name__ == "__main__":
    train_all_models()
