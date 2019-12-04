wxGeoMachines
=====================================================================

The final project of CSCI-470 Machine Learning course at Colorado School
of Mines. Developed by the GeoMachines group, this application is capable
of generating sonic logs using Machine Learning methods.

Two pre-trained models are available with the source code. The first
one uses Random Forest, and the second one uses a Feed-Forward Neural
Network (FFNN). There is also an implementation of a clustering method
to remove outliers based on DBSCAN.


Getting Started
---------------

These instructions will get you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on how to deploy the project on a live system.


Pre-requisites
---------------
In order to run this application you will need:

* Matplotlib   - Python 2D plotting library (https://matplotlib.org/)

* NumPy        - The fundamental package for scientific computing with
                 Python (https://numpy.org/)

* pandas       - BSD-licensed library providing high-performance, easy-
                 to-use data structures and data analysis tools for the
                 Python programming language.
                 (https://pandas.pydata.org/)

* SciPy        - Python-based ecosystem of open-source software for
                 mathematics, science, and engineering
                 (https://www.scipy.org/)

* Scikit-learn - Simple and efficient tools for predictive data analysis
                 (https://scikit-learn.org/stable/)

* TensorFlow   - End-to-end open-source platform for machine learning
                 (https://www.tensorflow.org/)

* wxPython     - The cross-platform GUI toolkit for the Python language.
                 (https://wxpython.org/)

* lasio        - Python package to read and write Log ASCII Standard
                 (LAS) files (https://lasio.readthedocs.io/)


Installing
---------------
The easiest way to fulfill the pre-requisites is to install Anaconda
(with Python 3) an, after that, to type:

$ conda install matplotlib numpy pandas scipy sklearn tensorflow wx

And:

$ pip install lasio


Running
---------------
To run the software in Windows and Linux systems, go to the software
main directory and type:

$ python wxGeomechine.py

For Mac users use:

$ pythonw wxGeomechine.py


Built With
---------------
wxPython - The cross-platform GUI toolkit for the Python language.
           (https://wxpython.org/)


Authors
---------------
Max Velasques (main GUI developer), Andrea Damasceno, Atilas Silva,
Samuel Chambers and Meng Jia


License
---------------
This project is licensed under the MIT License - see the LICENSE.txt
file for details

The LAS and ASC files included in the DATA directory were extracted
from the "Western Australian Petroleum and Geothermal Information
Management System" (WAPIMS) database. These files follow the rules
and regulations available on their website
(https://wapims.dmp.wa.gov.au/wapims).

