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
* `final_analysis` - Directory contains Jupyter notebooks used for final analysis.
  * `Final Analysis.ipynb` - Jupyter notebook with code for generating final parameter estimates and one
  plot of multiple country-level regressions.
  * `Final Plots.ipynb` - Jupyter notebook with code for generating plots from final parameter estimates.
* `final_output` - Directory contains graphs in `*.png` formats and Excel files used for manuscript
  * `Final_Output.xlsx` - Output from Jupyter notebook
  * `Final_Output_v2.xlsx` - Same data as original, modified to add columns with country rankings, 
  BIC difference values, etc. This is what is presented in the manuscript.
* `old_files` - Directory contains project files that were unrelated to the final analysis and output.
These include code for initial data exploration and model building.
