import os


def convert_xyz_files_to_sdf(xyz_folder_path: str, sdf_output_file_path: str):
    """
    Converts all XYZ files in a specified folder to a single SDF file.

    Parameters
    ----------
    xyz_folder_path: str
        Path to the folder containing XYZ files.
    sdf_output_file_path: str
        Path to the output SDF file to be created.

    Raises
    ------
    Exception: If openbabel module (pybel) is not found or not installed..
    """
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        raise Exception("XYZ converter requires openbabel to be installed.")

    output = pybel.Outputfile("sdf", sdf_output_file_path, overwrite=True)

    for filename in sorted(os.listdir(xyz_folder_path)):
        if filename.endswith(".xyz"):
            filepath = os.path.join(xyz_folder_path, filename)
            mol = next(pybel.readfile("xyz", filepath))
            output.write(mol)

    output.close()
