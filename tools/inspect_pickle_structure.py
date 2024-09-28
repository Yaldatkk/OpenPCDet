import argparse
import pickle

def load_pickle_file(file_path):
    """Load and inspect the pickle file structure."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Print the type and first few elements of the data to inspect
    print("Data type:", type(data))
    import time
    time.sleep(20)
    if isinstance(data, list):
        print("First element type:", type(data[0]))
        for idx, item in enumerate(data[:1]):  # Inspect the first 5 elements
            print(f"Item {idx}: {item}")
    elif isinstance(data, dict):
        print("Keys in the dictionary:", list(data.keys()))
        for key, value in data.items():
            print(f"Key: {key}, Value: {value}")
            break  # Only show the first key-value pair
    else:
        print("Unhandled data structure.")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Inspect pickle file')
    parser.add_argument('--pickle_file', type=str, required=True, help='Path to the pickle file')

    args = parser.parse_args()
    
    # Load the pickle file and inspect its structure
    load_pickle_file(args.pickle_file)

if __name__ == '__main__':
    main()

