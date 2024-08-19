import os
import sys
from abaqus import *
from abaqusConstants import *
from odbAccess import *
from textRepr import *
import __main__
import visualization
import xyPlot
import displayGroupOdbToolset as dgo

# extract_output.py

def extract_output(odb_path, output_dir):
    

    # Open the ODB file
    session.openOdb(name=odb_path)
    odb = session.odbs[odb_path]

    

    # Create XY data from history for displacement
    xy0 = session.XYDataFromHistory(name='displacement', odb=odb, 
        outputVariableName='Spatial displacement: U3 PI: C20-1 Node 1 in NSET SET-1', 
        steps=('Step-1',))

    # Create XY data from history for contact force
    xy1 = session.XYDataFromHistory(name='reaction-force', odb=odb, 
        outputVariableName='Reaction force: RF3 PI: C20-1 Node 1 in NSET SET-1', 
        steps=('Step-1',))

    # Write XY report to file
    job_name = os.path.splitext(os.path.basename(odb_path))[0]
    session.writeXYReport(fileName=os.path.join(output_dir, '{}.rpt'.format(job_name)), 
        xyData=(xy0, xy1))
    
    
if __name__ == "__main__":
    try:
        # Get the last two command line arguments
        odb_path = sys.argv[-2]
        output_dir = sys.argv[-1]
        
        extract_output(odb_path=odb_path, output_dir=output_dir)

    except IndexError:
        print("Error: Not enough arguments provided after '--'.")
