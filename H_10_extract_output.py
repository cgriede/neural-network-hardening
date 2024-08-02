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
        outputVariableName='Spatial displacement: U3 PI: H5-1 Node 1 in NSET INDENTER', 
        steps=('Step-1',))

    # Create XY data from history for contact force
    xy1 = session.XYDataFromHistory(name='reaction-force', odb=odb, 
        outputVariableName='Reaction force: RF3 PI: H5-1 Node 1 in NSET INDENTER', 
        steps=('Step-1',))

    # Write XY report to file
    job_name = os.path.splitext(os.path.basename(odb_path))[0]
    session.writeXYReport(fileName=os.path.join(output_dir, '{}.rpt'.format(job_name)), 
        xyData=(xy0, xy1))
    

    assembly = odb.rootAssembly
    
    battery_name = 'QUARTER-BATTERY-1'
    element_set_name = 'CONTACTELEMENTS'
    node_set_name = 'CONTACTNODES'

    instance = assembly.instances[battery_name]

    element_set = instance.elementSets[element_set_name]
    element_set = element_set.elements
    node_set = instance.nodeSets[node_set_name]
    node_set = node_set.nodes

    # Create a dictionary with node labels as keys
    node_dict = {node.label: node for node in node_set}


    #generate ordered label set to compare to elemnt node labels
    node_set_labels = set()

    # Iterate over the current nodes and change their keys to match their labels
    for node in node_set:
        new_key = node.label
        node_set_labels.add(new_key)


    for node_label in node_set_labels:
        #output displacement of nodes
        xy_u1 = session.XYDataFromHistory(
            name='U1 node: {}'.format(node_label), 
            odb=odb, 
            outputVariableName='Spatial displacement: U1 PI: QUARTER-BATTERY-1 Node {} in NSET CONTACTNODES'.format(node_label), 
            steps=('Step-1', ))
        xy_u2 = session.XYDataFromHistory(
            name='U2 node: {}'.format(node_label), 
            odb=odb, 
            outputVariableName='Spatial displacement: U2 PI: QUARTER-BATTERY-1 Node {} in NSET CONTACTNODES'.format(node_label), 
            steps=('Step-1', ))
        xy_u3 = session.XYDataFromHistory(
            name='U3 node: {}'.format(node_label), 
            odb=odb, 
            outputVariableName='Spatial displacement: U3 PI: QUARTER-BATTERY-1 Node {} in NSET CONTACTNODES'.format(node_label), 
            steps=('Step-1', ))
        
        job_name = os.path.splitext(os.path.basename(odb_path))[0]
        session.writeXYReport(fileName=os.path.join(output_dir, '{}.rpt'.format(job_name)),
            xyData=(xy_u1, xy_u2, xy_u3))

         # Open a file to write the element-node information

    with open(os.path.join(output_dir, '{}.nmp'.format(job_name)), 'w') as output_file:

        for element in element_set:
            
            element_label = element.label
            
            connectivity = set(element.connectivity)

            surface_nodes = connectivity & node_set_labels

            #generate "hashmap" for elements, nodes, coordinates
            for node_label in surface_nodes:
                
                node = node_dict[node_label]
                coords = node.coordinates
                coords = node.coordinates
                output_file.write("Element {}: Node {} (Initial Position: {})\n".format(element_label, node_label, coords))


            #extract PEEQ
            xy_peeq = session.XYDataFromHistory(
            name='PEEQ Elem: {}'.format(element_label), 
            odb=odb, 
            outputVariableName='Equivalent plastic strain: PEEQ PI: QUARTER-BATTERY-1 Element {} Int Point 1 in ELSET CONTACTELEMENTS'.format(element_label), 
            steps=('Step-1', ))

            #extract MISES eq
            xy_mises = session.XYDataFromHistory(
            name='MISES', 
            odb=odb, 
            outputVariableName='Mises equivalent stress: MISES PI: QUARTER-BATTERY-1 Element {} Int Point 1 in ELSET CONTACTELEMENTS'.format(element_label), 
            steps=('Step-1', ))

            #extract S11
            xy_s11 = session.XYDataFromHistory(
            name='S11', 
            odb=odb, 
            outputVariableName='Stress components: S11 PI: QUARTER-BATTERY-1 Element {} Int Point 1 in ELSET CONTACTELEMENTS'.format(element_label), 
            steps=('Step-1', ))

            #extract S22
            xy_s22 = session.XYDataFromHistory(
            name='S22', 
            odb=odb, 
            outputVariableName='Stress components: S22 PI: QUARTER-BATTERY-1 Element {} Int Point 1 in ELSET CONTACTELEMENTS'.format(element_label), 
            steps=('Step-1', ))

            #extract S33
            xy_s33 = session.XYDataFromHistory(
            name='S33', 
            odb=odb, 
            outputVariableName='Stress components: S33 PI: QUARTER-BATTERY-1 Element {} Int Point 1 in ELSET CONTACTELEMENTS'.format(element_label), 
            steps=('Step-1', ))

            #extract S12
            xy_s12 = session.XYDataFromHistory(
            name='S12',
            odb=odb,
            outputVariableName='Stress components: S12 PI: QUARTER-BATTERY-1 Element {} Int Point 1 in ELSET CONTACTELEMENTS'.format(element_label),
            steps=('Step-1', ))

            #extract S13
            xy_s13 = session.XYDataFromHistory(
            name='S13',
            odb=odb,
            outputVariableName='Stress components: S13 PI: QUARTER-BATTERY-1 Element {} Int Point 1 in ELSET CONTACTELEMENTS'.format(element_label),
            steps=('Step-1', ))

            #extract S23
            xy_s23 = session.XYDataFromHistory(
            name='S23',
            odb=odb,
            outputVariableName='Stress components: S23 PI: QUARTER-BATTERY-1 Element {} Int Point 1 in ELSET CONTACTELEMENTS'.format(element_label),
            steps=('Step-1', ))

            # Write XY report to file
            job_name = os.path.splitext(os.path.basename(odb_path))[0]
            session.writeXYReport(fileName=os.path.join(output_dir, '{}.rpt'.format(job_name)),
                xyData=(xy_peeq, xy_mises, xy_s11, xy_s22, xy_s33, xy_s12, xy_s13, xy_s23))
    
if __name__ == "__main__":
    try:
        # Get the last two command line arguments
        odb_path = sys.argv[-2]
        output_dir = sys.argv[-1]
        
        extract_output(odb_path=odb_path, output_dir=output_dir)

    except IndexError:
        print("Error: Not enough arguments provided after '--'.")
