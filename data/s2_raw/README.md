# Data from: Improving access to essential medicines via decision-aware machine learning

[https://doi.org/10.5061/dryad.h9w0vt4tw](https://doi.org/10.5061/dryad.h9w0vt4tw)

## Description of the data and file structure

* Data S1: list of facilities
* Data S2: consumption data for evaluation

- Data S3: supply data (added random noises to comply data privacy agreement)

* Data S4: same as Data S2. Consumption data for evaluation but include control products

- Data S5: population based demand for each facility across products

### Files and variables

#### File: S1.csv

**Description:** facility list

##### Variables

* facility_type: categorizing facilities as Community Health Centre (CHC), Community Health Post (CHP), Maternal and Child Health Post (MCHP), or Clinic. 
* hf_pk: facility unique ID
* district: larger administrative regions, comprising a total of 16 districts

#### File: S2.csv.zip

**Description:** consumption data for evaluation. 

**Variables**

* **hf_pk**: unique facility ID
* **name1**: product name
* **date**: record date
* **Inventory balance**: stockout, received, consumption, openBalance, closeBalance
* **Facility information**: facility type, latitude (lat), longitude (long), district
* **Quarter**: time period
* **productID**: product ID
* **Consumption statistics**: normAvg (average consumption by product), normStd (standard deviation of consumption by product)

#### File: S3.csv

**Description:** This data includes stock information across different time periods. Note that random noise was added to the data to comply with data privacy agreements.

##### Variables

* Item: medication name with dosage and unit specification
* Stock: stock quantity 
* Quarter: time period

#### File: S4_AlternativeData.csv.zip

**Description:** Same as S2.csv.zip but include control products' information.

#### File: S5_dfImp_popbased.csv

**Description:** Population based demand for each facility across products, which is used to run population-based imputation

**Variables**

* **quarterID**: time period ID
* **hf_pk**: unique facility ID
* **productID**: product ID
* **name1**: product name
* **Q3AIalloc**: machine-learning-based allocation decision in Q3
* **popDQ3**: estimated demand proportional to population size in Q3
* **Q2AIalloc**: machine-learning-based allocation decision in Q2
* **popDQ2**: estimated demand proportional to population size in Q2
* **ExcelAlloc**: allocation decision based on the Excel tool
* **popD**: estimated demand proportional to population size

## Code/software

All data are in csv format.

## Access information

Data was derived from the following sources:

* DHIS2 
* Grid3

