
def save_json(json_data, save_name="temp"):
    with open(f"./res/json/{save_name}.txt", "w") as f:
        for k, v in json_data.items():
            f.write(f"{k}:{v}\n")