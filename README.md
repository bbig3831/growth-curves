# ART growth curve modeling project

This repository contains the data and code I used for modeling the scale-up of antiretroviral therapy
(ART) in sub-Saharan Africa while working with Stéphane Verguet at the Harvard T.H. Chan School of 
Public Health (HSPH).

This code was all written before I had any experience with version control software or proper package
management. I have tried to clean things up to make it easier to understand, but the code will likely
not work right out of the box. At the very least, you will need to change directory references within
the Python code to match your own.

The directory structure is as follows
* `requirements.txt` - Standard package requirements for the project. Package versions may have
conflicting dependencies, please try installing an earlier version of a package if you run into
dependency problems.
* `source_data` - Directory contains source data for analysis. Includes ART coverage rates and population data.
* `archived` - Directory contains old code/data from first drafts of analysis.
* `final_scripts` - Directory contains scripts for downloading data, running curve fitting algorithms, and 
making manuscript graphs.
* `output_files` - Directory that contains Excel spreadsheets with results and graphs for manuscript.