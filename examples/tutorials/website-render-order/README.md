# Website_Render_Order


This folder contains CSV files that define the order in which tutorials are to be rendered on the DeepChem's website.

--Each file define the order and group tutorials for a specified section on the website.
--The files are named according to the section they define the order for to be displayed on the website
 
 
 # Example :
  - introduction-to-deepchem.csv this appears on the website as it is ordered over here.


#Adding your own tutorial

1. Save your tutorial notebook in `examples/tutorials/`.
2. Edit the appropriate CSV in this folder.
3. Add a new row with the `title` and `filename` as first line.
4. Commit both the tutorial notebook and CSV file in your PR.

# Common Checks
- Check for mismatch of title in csv and name of notebook .See that they are same .
-Check for the order of tutorials in the csv file.


