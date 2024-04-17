import json
import os
import argparse

def create_json_files(data_dir):
    # Ensure the directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Define the commands and the corresponding indices
    commands = {
        # "pick up the sharpie": list(range(0, 25)),
        "put the green target to the green box": list(range(100, 150)),
        # "reach to the blue target": list(range(50, 100))
    }

    # Iterate over each command and its indices
    for command, indices in commands.items():
        for idx in indices:
            # Define the filename for the episode
            episode_filename = os.path.join(data_dir, f"episode_{idx}.json")

            # Define the content of the episode
            episode_content = [
                {
                    "command": command,
                    "start_timestep": 0,
                    "end_timestep": 63 - 1,
                    "type": "instruction",
                }
            ]

            # Write the episode content to the file
            with open(episode_filename, "w") as file:
                json.dump(episode_content, file)

    print(f"Files have been created in {data_dir}")


# Define the main function to execute the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    
    args = vars(parser.parse_args())
    data_directory = args['dataset_dir']# input("Enter the path for the data directory: ")

    create_json_files(data_directory)
    
