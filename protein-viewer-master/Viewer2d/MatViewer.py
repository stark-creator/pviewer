
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser,DSSP

# from moleculardata import colors, colorsDSSP, rgba, resdict
resdict = {'SER':'np','THR':'np','GLN':'np','ASN':'np','TYR':'np','CYS':'np',
           'ALA':'nnp','VAL':'nnp','LEU':'nnp','ILE':'nnp','MET':'nnp','PRO':'nnp','PHE':'nnp','TRP':'nnp','GLY':'nnp',
           'ASP':'neg','GLU':'neg',
           'LYS':'pos','ARG':'pos','HIS':'pos'}

colors = {'H':'white',
          'C':'black', 'CA':'black','CB':'black','CG':'black','CZ':'black',
          'N':'blue',
          'O':'red',
          'F':'green','CL':'green',
          'BR':'brown',
          'I':'darkviolet',
          'HE':'turquoise', 'NE':'turquoise', 'AR':'turquoise','XE':'turquoise','KR':'turquoise',
          'P':'orange',
          'S':'yellow',
          'B':'salmon',
          'LI':'purple','NA':'purple','K':'purple','RB':'purple','CS':'purple',
          'BE':'darkgreen','MG':'darkgreen','Ca':'darkgreen','SR':'darkgreen','BA':'darkgreen','RA':'darkgreen',
          'TI':'grey',
          'FE':'darkorange',
          'np':'rebeccapurple',
          'nnp':'gray',
          'neg':'navy',
          'pos':'darkred'}

colorsDSSP = {'H':'red',
              'B':'brown',
              'E':'yellow',
              'G':'purple',
              'I':'pink',
              'T':'turquoise',
              'S':'green',
              '-':'gray'}

rgba = {'white':(1.0,1.0,1.0,1.0), 
        'black':(0.02,0.02,0.02,1.0),
        'blue':(0.0,0.0,1.0,1.0),
        'red':(1.0,0.0,0.0,1.0),
        'green':(0.13,0.78,0.0,1.0),
        'brown':(0.4,0.0,0.0,1.0),
        'darkviolet':(0.24,0.0,0.4,1.0),
        'turquoise':(0.0,0.78,0.84,1.0),
        'orange':(0.84,0.53,0.0,1.0),
        'yellow':(0.86,0.9,0.0,1.0),
        'salmon':(1.0,0.75,0.51,1.0),
        'purple':(0.35,0.0,0.59,1.0),
        'darkgreen':(0.0,0.35,0.0,1.0),
        'grey':(0.59,0.59,0.59,1.0),
        'darkorange':(0.86,0.45,0.0,1.0),
        'pink':(0.94,0.55,1.0,1.0),
        'rebeccapurple':(0.34,0.25,0.63,1.0),
        'gray':(0.75,0.75,0.75,1.0),
        'navy':(0.0,0.06,0.51,1.0),
        'darkred':(0.55,0.0,0.0,1.0)}

#Sacados de https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
radius = {'H':1.2,
          'C':1.7, 'CA':1.7,'CB':1.7,'CG':1.7,'CZ':1.7,
          'N':1.55,
          'O':1.52,
          'F':1.47,'CL':1.75,
          'BR':1.85,
          'I':1.98,
          'HE':1.40, 'NE':1.54, 'AR':1.88,'XE':2.16,'KR':2.02,
          'P':1.8,
          'S':1.8,
          'B':1.92,
          'LI':1.82, 'NA':2.27,'K':2.75,'RB':3.03,'CS':3.43,
          'BE':1.53,'MG':1.73,'Ca':2.31,'SR':2.49,'BA':2.68,'RA':2.83}

def restype(residue):
    try:
        return resdict[residue]
    except:
        return 'pink'

def color(atom):
    try:
        return colors[atom]
    except:
        return 'pink'

def cDSSP(prediction):
    return colorsDSSP[prediction]

def crgbaDSSP(prediction):
    return rgba[colorsDSSP[prediction]]

def crgba(color):
    return rgba[color]

def colorrgba(atom):
    try:
        return rgba[colors[atom]]
    except:
        return rgba['pink']

def vrad(atom):
    try:
        return radius[atom]
    except:
        return 1.5
from mpl_toolkits.mplot3d import Axes3D


class MatViewer(object):
    
    visualization_modes = ['cpk','backbone','aminoacid','dssp']
    
    def __init__(self, pdbdata, mode='cpk'):
        
        #Analyze pdb file
        self.parser = PDBParser(QUIET=True,PERMISSIVE=True)
        self.structure = self.parser.get_structure('model',pdbdata)
        
        #DSSP prediction
        self.model = self.structure[0]
        self.dssp = DSSP(self.model, pdbdata)
        
        #Mode selection
        if mode not in MatViewer.visualization_modes:
            raise Exception('Not recognized visualization mode %s' % mode)
        self.mode = mode
        
        #Make the plot
        if self.mode == 'cpk':
            self.cpk2d()
        elif self.mode == 'backbone':
            self.bb2d()
        elif self.mode == 'aminoacid':
            self.aa2d()
        elif self.mode == 'dssp':
            self.dssp2d()
    
    def cpk2d(self):
        """Draws atoms in a CPK colour palette"""
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for atom_type, color in colors.iteritems():
            atoms = [atom for atom in self.structure.get_atoms() if atom.get_id() == atom_type]
            coordinates = [atom.coord for atom in atoms]
            if not len(atoms)==0:
                x, y, z=zip(*coordinates)
                ax.scatter(x, y, z, c=color, marker='o')
                
        #Select the atoms that are not identified
        atoms_1 = [atom for atom in self.structure.get_atoms()]
        atoms_2 = [atom for atom in self.structure.get_atoms() if atom.get_id() in colors.keys()]
        atoms_pink = list(set(atoms_1)-set(atoms_2))
        
        coordinates_pink = [atom.coord for atom in atoms_pink]
        xp, yp, zp = zip(*coordinates_pink)
        ax.scatter(xp, yp, zp, c='pink', marker='o')
        
        ax.axis("off")
        plt.show()
    
    def bb2d(self):
        """Draws CA-N atoms linked by lines"""
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for chain in self.structure.get_chains():
            can_atoms = [atom for atom in chain.get_atoms() if atom.get_name() == 'CA' or atom.get_name() == 'N']
            can_coordinates = [atom.coord for atom in can_atoms]
            x,y,z=zip(*can_coordinates)
            ccolor = np.random.rand(3,1)
            
            ax.plot(x, y, z, c=ccolor, linewidth=2)
            ax.scatter(x, y, z, c=ccolor, marker='o')
        
        ax.axis("off")
        plt.show()
    
    def aa2d(self):
        """Draws atoms using a colour palette depending on the type of residue"""
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        #Select residues only if they're aminoacids
        for resname, residuetype in resdict.iteritems():
            residues = [residue for residue in self.structure.get_residues() if residue.get_resname() == resname]
            rescoord = []
            color = rgba(residuetype)
            
            for residue in residues:
                atoms = [atom for atom in residue.get_atoms()]
                coordinates = [atom.coord for atom in atoms]
                rescoord.append(np.array(coordinates))
                
            if len(rescoord)>1:
                rescoord = np.concatenate(rescoord)
                
            if not len(residues)==0:
                x, y, z =zip(*rescoord)
                ax.scatter(x, y, z, c=color, marker='o')

        #Select residues that are not aminoacids, skipping water
        residues_1 = [residue for residue in self.structure.get_residues() if residue.get_resname() != 'HOH']
        residues_2 = [residue for residue in self.structure.get_residues() if residue.get_resname() in resdict.keys()]
        
        residues_pink = list(set(residues_1)-set(residues_2))
        rescoordpink = []
        
        for residue in residues_pink:
            atomspink = [atom for atom in residue.get_atoms()]
            coordinatespink = [atom.coord for atom in atomspink]
            rescoordpink.append(np.array(coordinatespink))
        
        if len(rescoordpink)>1:
            rescoordpink = np.concatenate(rescoordpink)
        
        xp, yp, zp = zip(*rescoordpink)
        ax.scatter(xp, yp, zp, c='pink', marker='o')
        
        ax.axis("off")
        plt.show()

    def dssp2d(self):
        """Draw CA-N atoms linked by lines, coloured by their tertiary structure prediction"""
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        #Create the residue lists linked to their predictions
        residues = [residue for residue in self.structure.get_residues() if residue.get_resname() in resdict.keys()]
        struct3 = [self.dssp[key][2] for key in list(self.dssp.keys())]
        respred = zip(struct3,residues)
        
        #Create the point cloud depending on the prediction
        for prediction, color in colorsDSSP.iteritems():
            residuesp = [residue[1] for residue in respred if residue[0] == prediction]
            
            predcoord_can = []
            for residue in residuesp:
                atomsp = [atom for atom in residue.get_atoms() if atom.get_name() == 'CA' or atom.get_name() == 'N']
                coordinatesp = [atom.coord for atom in atomsp]
                predcoord_can.append(np.array(coordinatesp))
                
            if len(predcoord_can)>1:
                predcoord_can = np.concatenate(predcoord_can)
                
            if not len(residuesp)==0:
                x, y, z = zip(*predcoord_can)
                ax.scatter(x, y, z, c=color, marker='o')
            
        #Create the chains linking the aminoacids
        for chain in self.structure.get_chains():
            can_atoms = [atom for atom in chain.get_atoms() if atom.get_name() == 'CA' or atom.get_name() == 'N']
            can_coordinates = [atom.coord for atom in can_atoms]
            x, y, z = zip(*can_coordinates)
            ccolor = np.random.rand(3,1)
            ax.plot(x, y, z, c=ccolor, linewidth=1)
            
        ax.axis("off")
        plt.show()