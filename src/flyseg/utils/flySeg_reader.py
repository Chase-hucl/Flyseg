import os
import h5py
import numpy as np
import nibabel as nib
import tifffile as tiff

def file_read(file_path):
    """
        Load a 3D volume from a file. Supported formats: .h5/.hdf5, .nii/.nii.gz, .tif/.tiff.

        Parameters:
            file_path (str): Path to the input file.

        Returns:
            np.ndarray: Loaded 3D volume as a NumPy array.

        Raises:
            ValueError: If the file format is unsupported or the file does not contain a valid 3D dataset.
        """
    if not isinstance(file_path, str):
        raise ValueError("Input must be a file path (string).")

    data = None

    # Handle HDF5 files
    if file_path.endswith(('.h5', '.hdf5')):
        with h5py.File(file_path, 'r') as h5_file:
            for name, obj in h5_file.items():
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                    data = obj[:]
                    break
            if data is None:
                raise ValueError("No suitable 3D dataset found in the HDF5 file.")

    # Handle NIfTI files
    elif file_path.endswith(('.nii', '.nii.gz')):
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()  # Returns data as float64 by default

    # Handle TIFF files
    elif file_path.endswith(('.tif', '.tiff')):
        data = tiff.imread(file_path)
        if data.ndim != 3:
            raise ValueError(f"Loaded TIFF file has shape {data.shape}, but only 3D data is supported.")

    else:
        raise ValueError("Unsupported file format. Supported formats: .h5, .nii, .nii.gz, .tif, .tiff.")

    return data

def data_save(data, save_dir, save_name, file_format='nii', affine=None):
    """
    Save a 3D volume to disk in NIfTI (.nii), HDF5 (.h5), or TIFF (.tif) format.

    Parameters:
    data (np.ndarray): 3D NumPy array representing the volume to save.
    save_dir (str): Directory where the file will be saved.
    save_name (str): Name of the saved file, including extension.
    file_format (str): Format to save the file in ('nii', 'h5', or 'tif'). Default is 'nii'.
    affine (np.ndarray, optional): Affine transformation matrix for NIfTI files. Defaults to identity matrix.

    Raises:
    ValueError: If an unsupported file format is specified.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    if file_format == 'nii':
        if affine is None:
            affine = np.eye(4)
            nii_image = nib.Nifti1Image(data, affine)
            nib.save(nii_image, save_path)

    elif file_format == 'h5':
        with h5py.File(save_path, 'w') as h5_file:
            h5_file.create_dataset('dataset1', data=data)

    elif file_format == 'tif':
        tiff.imwrite(save_path, data, imagej=True)

    else:
        raise ValueError(f"Unsupported file format: {file_format}. Supported formats: 'nii', 'h5', 'tif'.")

if __name__ == "__main__":
    file_path = r"X:\63148_annotation"
    data = file_read(file_path)
    save_dir = r"X:\63148_annotation\good"
    save_name = "test_10"
    data_save(data, save_dir, save_name, file_format='nii', affine=None)