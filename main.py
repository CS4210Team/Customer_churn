from project.train_all import train_all_models
from project.evaluate_all import evaluate_all_models

def main():
    print("Choose an option:")
    print("  1. Train models only")
    print("  2. Evaluate models only")
    print("  3. Train AND Evaluate (default)")
    
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        print("\n TRAINING MODE \n")
        train_all_models()

    elif choice == "2":
        print("\n EVALUATION MODE \n")
        evaluate_all_models()

    else:
        print("\n TRAINING THEN EVALUATION \n")
        train_all_models()
        evaluate_all_models()

if __name__ == "__main__":
    main()
