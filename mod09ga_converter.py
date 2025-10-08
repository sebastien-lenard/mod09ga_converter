"""
Develop code that creates reflectance, quality inputs, cloud and other inputs necessary for both STC and SPIReS that can be used for legacy MOD09GA v006 or more recent
MOD09GA v061.

Use example at the bottom of the script.
"""
import logging
from logging import Logger
from dotenv import find_dotenv, dotenv_values
import os
from pathlib import Path
import json
import argparse

from pyhdf.SD import SD, SDC
import numpy as np

class Mod09gaConverter:
    equivalent_hdf4_types_for_types = {
                        "str": SDC.CHAR,
                        "int8":  SDC.INT8,
                        "uint8": SDC.UINT8,
                        "int16": SDC.INT16,
                        "uint16": SDC.UINT16,
                        "int32": SDC.INT32,
                        "uint32": SDC.UINT32,
                        "int64": SDC.UINT32,
                        "float32": SDC.FLOAT32,
                        "float64": SDC.FLOAT64
                    }
    # Warning. HDF 4 doesnt support int64. Force here to uint32 (only occurs for
    # variable QC_500m_1, but makes code may not working for other variables with
    # high range of values)

    def __init__(
            self,
            env_filepath: str | None = None,
            conf_filepath: str | None = None,
            verbosity_level: int = logging.INFO
            ):
        """
        Parameters
        ----------
        env_filepath: filepath of the .env configuration file. Default: .env in the
            project is used.
        conf_filepath: filepath of the configuration of variables to convert. Default,
            the filepath configured in .env
        verbosity_level: Level of logging, logging.DEBUG for more verbosity
            or logging.INFO for less.
        """
        # Log instantiation
        ################################################################################
        self.logger = Logger(self.__class__.__name__)
        
        # Console handler
        handler = logging.StreamHandler()
        handler.setLevel(verbosity_level)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.setLevel(verbosity_level)

        ################################################################################

        self.logger.debug(f"Instantiates Mod09gaConverter...")

        # Env and configuration
        ################################################################################
        self.env_filepath = env_filepath if env_filepath is not None else find_dotenv()
        self.env = dotenv_values(self.env_filepath)
        self.data_root_directory_path = os.path.expandvars(
            self.env.get("SCRATCH_DATA_DIRECTORY"))
            # e.g. /scratch/XXXX/
        
        self.conf_filepath = conf_filepath if conf_filepath is not None else \
            os.path.expandvars(
                self.env.get("DEFAULT_MOD09GA_CONVERTER_CONFIGURATION_FILEPATH"))
        try:
            with open(self.conf_filepath, 'r') as f:
                self.conf = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at '{self.conf_filepath}'")
            raise
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{self.conf_filepath}': {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading '{self.conf_filepath}': {e}")
            raise
        self.logger.debug(f"Instantiated Mod09gaConverter.")

    def delete_file(
            self,
            file_path: str
            ):
        """
        Parameters
        ----------
        file_path: Absolute filepath to delete.
        """
        this_file = Path(file_path)
        if this_file.exists():
            self.logger.debug(f"Deleting {file_path}...")
            try:
                os.remove(this_file)
                self.logger.debug(f"Deleted {file_path}.")
            except OSError as e:
                self.logger.error(f"Error deleting {file_path}: {e}")
                raise
        
    def lower_precision(
            self,
            input_file_path: str,
            output_file_path: str
            ):
        """
        Generates the lower precision version of a MOD09GA.006 or MOD09GA.061 file.

        Parameters
        ----------
        input_file_path: Absolute filepath of the mod09ga file to convert into a file of
            lower precision. File should be MOD09GA.006 or MOD09GA.061 .hdf format
            (HDF4 or HDF-EOS2).
        output_file_path: Absolute filepath where the lower precision file is saved.
            Should have .hdf extension.
        """
        self.logger.info(f"Starts Mod09gaConverter.lower_precision of "
                          f"{input_file_path} into {output_file_path}...")

        directory_path = Path(output_file_path).parent
        # Create the directory and any necessary parent directories
        directory_path.mkdir(parents=True, exist_ok=True)

        # For HDF-EOS, safest to create a new file to avoid internal structure errors.
        self.delete_file(output_file_path)

        hdf_in = None
        hdf_out = None

        try:
            self.logger.debug("Opening input file...")
            hdf_in = SD(input_file_path, SDC.READ)

            self.logger.debug("Creating output file...")
            hdf_out = SD(output_file_path, SDC.CREATE | SDC.WRITE)
                
            self.logger.debug("Copy global attributes (in string for simplification)"
                              "...")
            try:
                global_attributes = hdf_in.attributes()

                for attr_name, attr_value in global_attributes.items():
                    attr = hdf_out.attr(attr_name)

                    if isinstance(attr_value, (list, tuple, np.ndarray)):
                        separator = ", "
                        attr_value = separator.join(map(str, attr_value))
                    elif isinstance(attr_value, (int, float)):
                        attr_value = str(attr_value)
                    elif not isinstance(attr_value, str):
                        self.logger.error(f"Unknown global attribute type for '"
                                          f"{attr_name}': {type(attr_value)}")
                        raise
                    
                    attr.set(SDC.CHAR, attr_value)

                self.logger.debug("Copied global attributes.")

            except Exception as e:
                self.logger.error(f"Error copying global attributes: {e}",
                                  exc_info=True)
                raise

            # List and loop through the SDS datasets to convert
            input_datasets_dict = hdf_in.datasets()
            for dataset_name in self.conf["file_variables"].keys():
                if dataset_name not in input_datasets_dict:
                    self.logger.warning(f"Dataset '{dataset_name}' not found in input"
                                        f" file. Skipping.")
                    continue

                self.logger.debug(f"Reading: {dataset_name}...")
                sds_in = hdf_in.select(dataset_name)
                data = sds_in.get()

                input_file_attributes = sds_in.attributes()

                # Correction of an incorrect scale_factor for MOD09GA.006 reflectances.
                if "scale_factor" in input_file_attributes.keys() and \
                    input_file_attributes["scale_factor"] == 10000:
                    input_file_attributes["scale_factor"] = 0.0001

                sds_info = sds_in.info()
                    # tuple. E.g. ('state_1km_1', 2, [1200, 1200], 23, 6) with
                    # information at position 0: dataset name= 'state_1km_1', 
                    # 1: number of dimensions = 2, 2: shape = [1200, 1200],
                    # 3: type =  SDC.UINT16 (23), 4: number of attributes = 6.

                data_type = sds_info[3] 
                dim_sizes = sds_info[2]
                # Dimension names for HDF-EOS compatibility.
                dim_names = [sds_in.dim(i).info()[0] for i in range(sds_info[1])]

                sds_in.endaccess()
 
                self.logger.debug(f"Creating: {dataset_name}, dimensions and type are"
                                  f" extracted from the original dataset...")
                sds_out = hdf_out.create(dataset_name,
                                         data_type,
                                         tuple(dim_sizes))
                
                try:
                    # Compression level (1-9, where 9 is max compression)
                    DEFLATE_LEVEL = 5
                    sds_out.setcompress(SDC.COMP_DEFLATE, DEFLATE_LEVEL)
                    self.logger.debug(f"Applied GZIP level 5 compression to '"
                                      f"{dataset_name}'.")
                except Exception as e:
                    self.logger.error(f"Failed to set compression for '{dataset_name}'"
                                      f": {e}.")
                    raise

                # Copy dimension names (essential for HDF-EOS compatibility)
                for i, dim_name in enumerate(dim_names):
                    sds_out.dim(i).setname(dim_name)

                self.logger.debug(f"Get input/ouput attributes, minimal check of input"
                                  f" attributes: {dataset_name}...")
                dataset_conf = self.conf["file_variables"][dataset_name]

                if input_file_attributes["_FillValue"] != \
                    dataset_conf["input"]["nodata"] or \
                        input_file_attributes["valid_range"][0] != \
                            dataset_conf["input"]["min"] or \
                    input_file_attributes["valid_range"][1] != \
                        dataset_conf["input"]["max"] or \
                    ("scale_factor" in dataset_conf["input"].keys() and \
                     input_file_attributes["scale_factor"] != \
                     dataset_conf["input"]["scale_factor"]):
                    self.logger.error(f"Attributes of {dataset_name} in file "
                                      f"{input_file_path} do not correspond to "
                                      f"attributes in conf {self.conf_filepath}.")
                    raise

                self.logger.debug(f"Convert data into lower resolution if required by"
                                  f" conf: {dataset_name}...")
                if "output" in dataset_conf.keys():
                    data = np.float64(data)
                    
                    data[data == dataset_conf["input"]["nodata"]] = np.nan
                    if "scale_factor" in dataset_conf["input"].keys() and \
                        "scale_factor" in dataset_conf["output"].keys():
                        data = data * dataset_conf["input"]["scale_factor"] / \
                            dataset_conf["output"]["scale_factor"]
                    data[data < dataset_conf["output"]["min"]] = np.nan
                    data[data > dataset_conf["output"]["max"]] = np.nan
                    data = np.round(data, decimals=0)
                    data[np.isnan(data)] = dataset_conf["output"]["nodata"]
                    data = data.astype(np.dtype(dataset_conf["output"]["type"]))
                self.logger.debug(f"Writing data: {dataset_name}...")
                sds_out[:] = data

                self.logger.debug(f"Writing attributes: {dataset_name}...")

                for attr_name, attr_value in input_file_attributes.items():

                    # Skip output attributes that changed and will be written later
                    if "output" in dataset_conf.keys() and \
                        attr_name in ("_FillValue", "valid_range", "scale_factor"):
                        continue

                    attr_type = type(attr_value).__name__

                    if attr_name in ("_FillValue", "valid_range"):
                        attr_type = data_type

                    else:
                        if attr_type in ("list", "tuple", "np.ndarray", "int", 
                                         "float", "np.integer", "np.floating"):
                            attr_type = np.array(attr_value).dtype.type.__name__

                        if attr_type in self.equivalent_hdf4_types_for_types.keys():
                            attr_type = self.equivalent_hdf4_types_for_types[attr_type]
                        else:
                            self.logger.warning(f"Skipping attribute {attr_name} with "
                                                f"unknown type: {attr_type}")
                            continue

                    sds_out.attr(attr_name).set(attr_type, attr_value)

                if "output" in dataset_conf.keys():
                    self.logger.debug(f"Writing changed attributes (_FillValue, range, "
                                    f"scale_factor): {dataset_name}...")
                    if dataset_conf["output"]["type"] not in \
                        self.equivalent_hdf4_types_for_types.keys():
                        self.logger.error(f"Type of {dataset_name} in conf "
                                        f"{self.conf_filepath} is not found in the "
                                        f"equivalence list of hdf4 types.")
                        raise
                    hdf4_output_type = self.equivalent_hdf4_types_for_types[ \
                        dataset_conf["output"]["type"]]
                    sds_out.attr("valid_range").set(
                        hdf4_output_type,
                        [dataset_conf["output"]["min"], dataset_conf["output"]["max"]])
                    sds_out.attr("_FillValue").set(
                        hdf4_output_type, dataset_conf["output"]["nodata"])
                    sds_out.attr("scale_factor").set(
                        SDC.FLOAT32, dataset_conf["output"]["scale_factor"])

                sds_out.endaccess()

            self.logger.info(f"Achieved Mod09gaConverter.lower_precision of "
                          f"{input_file_path} into {output_file_path}...")

        except Exception as e:
            self.logger.error(f"An error occurred: {e}", exc_info=True)
            # Delete output file unless debug
            if self.logger.getEffectiveLevel() != logging.DEBUG:
                self.logger.warning(f"Delete file {output_file_path}...")    
                self.delete_file(output_file_path)
            raise

        finally:
            # Close the HDF files
            if hdf_in:
                hdf_in.end()
                self.logger.debug("Input HDF file closed.")
            if hdf_out:
                hdf_out.end()
                self.logger.debug("Output HDF file closed.")

    def lower_precision_for_directory(
            self,
            input_directory_path: str,
            output_directory_path: str,
            delete_previous_output: bool | None = False
            ):
        """
        Scans an input directory path, and convert all the MOD09GA.006 and MOD09GA.061
            files into lower precision files of same name saved in a output directory.
        
        Parameters
        ----------
        input_directory_path: Absolute path of the directory hosting the mod09ga files
            to convert into files of lower precision. Files should be MOD09GA.006 or MOD09GA.061 .hdf format
            (HDF4 or HDF-EOS2).
        output_directory_path: Absolute path of the directory where the lower precision
            filese are saved.
        delete_previous_output: Default False. Keeps the previous files, which allows to
            skip previously generated output files. True: delete all previous files of
            the output directory.
        """

        self.logger.info(f"Starts Mod09gaConverter.lower_precision_for_directory of "
                          f"{input_directory_path} to {output_directory_path}...")

        input_directory = Path(input_directory_path)
        output_directory = Path(output_directory_path)

        output_directory.mkdir(parents=True, exist_ok=True)

        # 2. Delete existing files if forced
        if delete_previous_output == True:
            self.logger.warning(f"Delete files in {output_directory_path}...")
            
            for file_path in output_directory.glob('*'):
                if file_path.is_file():
                    os.remove(file_path)
            self.logger.info(f"Deleted files in {output_directory_path}.")

        # Match MOD09GA input files and loop to convert if output doesnt exist
        for search_pattern in ["MOD09GA*.006*.hdf", "MOD09GA*.061*.hdf"]:
            for input_file_path_obj in input_directory.glob(search_pattern):
                # Absolute path
                input_file_path = str(input_file_path_obj.resolve()) 
                
                # Relative path
                relative_path = input_file_path_obj.relative_to(input_directory)
                
                # Output path
                output_file_path_obj = output_directory / relative_path
                output_file_path = str(output_file_path_obj)

                # 3. Check for file existence in the output directory
                if output_file_path_obj.exists():
                    self.logger.info(f"Skip existing {relative_path}.")
                    skipped_count += 1
                else:
                    self.lower_precision(input_file_path, output_file_path)

        self.logger.info("-" * 40)
        self.logger.info(f"Achieved Mod09gaConverter.lower_precision_for_directory of "
                          f"{input_directory_path} to {output_directory_path}...")
        self.logger.info("-" * 40)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Conversion of MOD09GA.")

    parser.add_argument("--input", type=str, required=True,
                        help="input file path or input directory")
    parser.add_argument("--output", type=str, required=True,
                        help="output file path or output directory")
    parser.add_argument("--delete", type=bool, default=False,
                        help="Delete previous output files (always delete when only one file), default False")
    parser.add_argument("--verbosity", type=int, default=logging.INFO,
                        help="Log verbosity, default logging.INFO")

    args = parser.parse_args()

    mod09ga_converter = Mod09gaConverter(verbosity_level = args.verbosity)

    if os.path.isfile(args.input):
        mod09ga_converter.lower_precision(
            args.input,
            args.output,
            )
    elif os.path.isdir(args.input):
        mod09ga_converter.lower_precision_for_directory(
            args.input,
            args.output,
            delete_previous_output=args.delete
            )
    
    """
    E.g. of input / output files:
    input = mod09ga_converter.data_root_directory_path + "/input/mod09ga.061/h08v05/2023/MOD09GA_061-20250912_010844/MOD09GA.A2023091.h08v05.061.2023115073625.hdf"
    output = mod09ga_converter.data_root_directory_path + "/output/mod09ga.061/h08v05/2023/MOD09GA_061-20250912_010844/MOD09GA.A2023091.h08v05.061.2023115073625.hdf
    
    E.g. of input / output directories:
    input = mod09ga_converter.data_root_directory_path + "/input/mod09ga.061/h08v05/2023/MOD09GA_061-20250912_010844/",
    output = mod09ga_converter.data_root_directory_path + "/output/mod09ga.061/h08v05/2023/MOD09GA_061-20250912_010844/"
    """
