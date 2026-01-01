import json

# Just check the train.json file
with open("training_splits/train.json", 'r') as f:
    data = json.load(f)

print("Type of loaded data:", type(data))
print("Keys if dict:", list(data.keys()) if isinstance(data, dict) else "Not a dict")

if isinstance(data, dict):
    if 'data' in data:
        actual_data = data['data']
        print("Type of data['data']:", type(actual_data))
        print("Length of data['data']:", len(actual_data))
        print("First item in data['data']:", actual_data[0] if len(actual_data) > 0 else "Empty")
    else:
        print("No 'data' key found")
        print("All keys:", list(data.keys()))
elif isinstance(data, list):
    print("Direct list length:", len(data))
    print("First item:", data[0] if len(data) > 0 else "Empty")