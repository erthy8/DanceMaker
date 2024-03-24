import os


def create_folder_if_not_exists(folder_name):
    try:
        # Create the new folder path
        os.makedirs("data/" + folder_name)
    except FileExistsError:
        # Directory already exists
        pass
