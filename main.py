from project.train import train_models
from project.evaluate import evaluate_models

def main():
    print("Choose an option:")
    print("  1. Train models only")
    print("  2. Evaluate models only")
    print("  3. Train AND Evaluate (default)")
    
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        print("\n TRAINING MODE \n")
        train_models()

    elif choice == "2":
        print("\n EVALUATION MODE \n")
        evaluate_models()

    else:
        print("\n TRAINING THEN EVALUATION \n")
        train_models()
        evaluate_models()

if __name__ == "__main__":
    main()
