from project.model_utils.logreg.evaluate import evaluate_logreg
from project.model_utils.tree.evaluate import evaluate_tree
from project.model_utils.knn.evaluate import evaluate_knn
from project.model_utils.svm.evaluate import evaluate_svm

def evaluate_all_models():
    print("===== EVALUATING ALL MODELS =====\n")
    
    evaluate_logreg()
    print("\n-------------------------------\n")
    
    evaluate_tree()
    print("\n-------------------------------\n")
    
    evaluate_knn()
    print("\n-------------------------------\n")
    
    evaluate_svm()
    print("\n===== ALL MODELS EVALUATED =====")

if __name__ == "__main__":
    evaluate_all_models()
