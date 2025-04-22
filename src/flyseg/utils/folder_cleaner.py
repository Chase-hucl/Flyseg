import os
import shutil
import logging

def clear_folder(folder_path: str, remove_subdirs: bool = True, verbose: bool = True):
    """
    Clears the contents of a specified folder, including files and optionally subdirectories,
    without deleting the folder itself.

    Args:
        folder_path (str): The path to the folder to clear.
        remove_subdirs (bool): If True, subdirectories will also be removed. Defaults to True.
        verbose (bool): If True, logs details of the operation. Defaults to True.

    Raises:
        ValueError: If the folder path does not exist.
    """
    if not os.path.exists(folder_path):
        msg = f"⚠️ Folder '{folder_path}' does not exist. Skipping cleanup."
        if verbose:
            print(msg)
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                if verbose:
                    print(f"🗑️ Removed file: {file_path}")
            elif os.path.isdir(file_path):
                if remove_subdirs:
                    shutil.rmtree(file_path, ignore_errors=True)
                    if verbose:
                        print(f"🗂️ Removed folder: {file_path}")
        except PermissionError as e:
            print(f"❌ Permission denied: {file_path} - {e}")
        except OSError as e:
            print(f"❌ Failed to delete {file_path}: {e}")

    if verbose:
        print(f"✅ Folder '{folder_path}' cleared successfully.")
