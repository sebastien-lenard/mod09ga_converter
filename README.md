## Install and run

This python code converts MOD09GA.006 or MOD09GA.061 files into files restricted to variables necessary to the STC and SPIReS code, with lower precision. 

The variables handled and precision types are configured in the mod09ga_converter.json file.
The .env file contains basic configuration of paths.

To run the project, it is advised to create and use a dedicated mamba environment, after cd to the root of the project;

```
mamba env create -n mod09ga_converter_env -f environment.yml
mamba activate mod09ga_converter_env
```

Then the user can run the code on 1 file: on a directory containing the MOD09GA files.
```
mamba activate mod09ga_converter_env
input=${slurmScratchDir1}"/input/mod09ga.061/h08v05/2023/MOD09GA_061-20250912_010844/MOD09GA.A2023091.h08v05.061.2023115073625.hdf"
output=${slurmScratchDir1}"/output/mod09ga.061/h08v05/2023/MOD09GA_061-20250912_010844/MOD09GA.A2023091.h08v05.061.2023115073625.hdf"
verbosity=10 # logging.DEBUG

python mod09ga_converter.py --input ${input} --output ${output} --verbosity=${verbosity}
```

or on one directory:
```
mamba activate mod09ga_converter_env
input=${slurmScratchDir1}"/input/mod09ga.061/h08v05/2023/MOD09GA_061-20250912_010844/"
output=${slurmScratchDir1}"/output/mod09ga.061/h08v05/2023/MOD09GA_061-20250912_010844/"
delete=True # delete previous output if exists

python mod09ga_converter.py --input ${input} --output ${output} --delete=${delete}
```

## Information about the output files
The output files are of the same type as the input files, the deprecated format HDF4 - HDF-EOS2. All the variables are stored as SDS datasets, and the hdf internal structure by 500-m and 1-km is not kept, for simplicity.

The global attributes are preserved as string for simplicity too.

The variable attributes are preserved, in particular the names of datasets (e.g. sur_refl_b01_1 for surface reflectance band 1, see mod09ga_converter.json conf file) and the dimensions (2400x2400 for 500-m datasets or 1200x1200 for 1-km datasets), except for the attributes _FillValue, valid_range and scale_factor, which are updated to the lower precision values (for all variables except state_1km_1, QC_500m_1, and num_observations_500m).

For instance, reflectances are set to 255 value for any input value outside of the 0-16000 range, and then scaled with a factor of 0.01, to fit the range 0-160 and be saved as uint8 type.

The variables state_1km_1, QC_500m_1, and num_observations_500m are preserved and not saved in lower precision (changing precision for state_1km_1 and QC_500m_1 has a big impact because information is saved as bits in these variables).

## Use for STC and Spires.

### STC

To ingest the new files for STC, there's the need to create a concurrent conf/MOD09GA_cstruct.mat file. This updated version should restrict itself to the variables in mod09ga_converter.json and have the fields precision, minimum, maximum, scale, fill updated to the values in mod09ga_converter.json.

Checks and adaptations of the scripts readMOD09day.m (some angle info are hard-coded) and scale_MOD09GA.m are probably necessary.

### SPIReS v2024.1.0

To ingest the new files for SPIReS v2024.1.0, the script fill_and_run_modis20240204.m needs to have updated values for dtype and divisor. A check on fillMODIScube20240204.m is also necessary. Additionally, the original SPIReS function GetMOD09GA() and other functions called in it might need to be modified. These scripts are in the SPIReS version of Ned and used by SPIReS v2024.1.0.

### SPIReS v2025.0.1

To ingest the new files for SPIReS v2025.0.1, an update of the lines for mod09ga of conf/configuration_of_versionvariables.csv is necessary, to update the nodata, type, min, max. A check of SpiresInversor.m is also necessary for the reflectances, as there are used in a specific way by the cloud neural network.
