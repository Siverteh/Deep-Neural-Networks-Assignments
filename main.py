from assignments.assignment_1.k_nearest_neighbour import KNN
from assignments.assignment_2.neural_networks import MLPModels  # Assuming you save the MLPModels class in mlp_models.py

def run_knn():
    while True:
        max_k = input("Enter max k for K Nearest Neighbour: ")

        try:
            max_k = int(max_k)  # Convert the input to an integer
            knn = KNN()
            knn.run_knn(max_k)
            break  # Exit the loop after running the KNN function
        except ValueError:
            print("Invalid choice. Please enter a valid whole number.")

def run_simple_mlp():
    dataset_path = '/workspaces/Small_Assignments/datasets/ecoli/ecoli.data'
    mlp_models = MLPModels(dataset_path)
    mlp_models.run_simple_mlp()

def run_keras_mlp():
    dataset_path = '/workspaces/Small_Assignments/datasets/ecoli/ecoli.data'
    mlp_models = MLPModels(dataset_path)
    mlp_models.run_keras_mlp()

def menu():
    print("Select an assignment to run:")
    print("1: K-Nearest Neighbour")
    print("2: Simple MLP (No Training)")
    print("3: Keras MLP (Training Included)")
    print("0: Exit")

def main():
    functions = {
        "1": run_knn,
        "2": run_simple_mlp,
        "3": run_keras_mlp,
    }

    while True:
        menu()
        choice = input("Enter your choice: ")

        if choice == "0":
            print("Exiting the program.")
            break
        elif choice in functions:
            functions[choice]()
        else:
            print("Invalid choice. Please enter a valid number.")

if __name__ == "__main__":
    main()
