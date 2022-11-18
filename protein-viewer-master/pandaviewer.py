from direct.showbase.ShowBase import ShowBase
from panda3d.core import ColorAttrib, TransparencyAttrib, AntialiasAttrib
from panda3d.core import LVecBase4f
from panda3d.core import NodePath, PandaNode, TextNode
from panda3d.core import AmbientLight, DirectionalLight
from direct.task import Task
from panda3d.core import LineSegs
from panda3d.core import Mat4

from Bio.PDB import PDBParser,DSSP

from math import pi, sin, cos
import numpy as np

# from molecular_data import resdict, restype, colorrgba, vrad, crgbaDSSP
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

import sys, os

class PandaViewer(ShowBase):
    
    if len(sys.argv[1:])!=1:
        raise SystemExit("Uso: %s Archivo_PDB" % os.path.basename(sys.argv[0]))
    
    def __init__(self):
        ShowBase.__init__(self)
        self.cloud = False
        self.help = False
        self.screen_text = []
        
        #Desglosamos archivo PDB
        pdbdata = sys.argv[1]
        parser = PDBParser(QUIET=True,PERMISSIVE=True)
        structure = parser.get_structure('model', pdbdata)
        
        #Hacemos la prediccion DSSP
        model = structure[0]
        dssp = DSSP(model, pdbdata)
        
        #Creamos los modelos
        self.cpknode = render.attachNewNode("CPK")
        self.aanode = render.attachNewNode("Aminoacids")
        self.bbnode = render.attachNewNode("BackBone")
        self.dsspnode = render.attachNewNode("DSSP")
        self.nnode = render.attachNewNode("Cloud")
        
        #CPK
        for atom in structure.get_atoms():
            x, y, z = atom.coord
            atomid = atom.get_id()
            a = loader.loadModel("data/atom_sphere")
            a.setPos(x, y, z)
            a.reparentTo(self.cpknode)
            a.setColor(colorrgba(atomid))
            a.setScale(vrad(atomid))

        self.cpknode.flattenStrong()
            
        #Aminoacids
        self.residues = [residue for residue in structure.get_residues() if residue.get_resname() in resdict.keys()]
        for residue in self.residues:
            resid = residue.get_resname()
            color = colorrgba(restype(resid))
            atoms = [atom for atom in residue.get_atoms()]
            for atom in atoms:
                x, y, z=atom.coord
                atomid=atom.get_id()
                a = loader.loadModel("data/atom_sphere")
                a.setPos(x, y, z)
                a.setColor(color)
                a.setScale(vrad(atomid))
                a.reparentTo(self.aanode)

        self.residues2 = [residue for residue in structure.get_residues() if not residue in self.residues and residue.get_resname() != 'HOH']
        for residue in self.residues2:
            atoms = [atom for atom in residue.get_atoms()]
            for atom in atoms:
                x, y, z=atom.coord
                atomid=atom.get_id()
                a = loader.loadModel("data/atom_sphere")
                a.setPos(x, y, z)
                a.setColor(colorrgba(atomid))
                a.setScale(vrad(atomid))
                a.reparentTo(self.aanode)
        self.aanode.flattenStrong()
        self.aanode.hide()
        
        #Backbone
        for chain in structure.get_chains():
            carr = np.random.rand(3,1)
            ccolor = float(carr[0]),float(carr[1]),float(carr[2]),1.0
            can_atoms = [atom for atom in chain.get_atoms() if atom.get_name() == 'CA' or atom.get_name() == 'N']
            can_coordinates = [atom.coord for atom in can_atoms]
            for atom in can_atoms:
                x, y, z = atom.coord
                atomid=atom.get_id()
                a = loader.loadModel("data/atom_sphere")
                a.setPos(x,y,z)
                a.reparentTo(self.bbnode)
                a.setColor(ccolor)
                a.setScale(vrad(atomid)/2.5)

            lines = LineSegs()
            lines.setColor(ccolor)
            lines.moveTo(can_coordinates[0][0],can_coordinates[0][1],can_coordinates[0][2])
            for i in range(len(can_atoms))[1:]:
                lines.drawTo(can_coordinates[i][0],can_coordinates[i][1],can_coordinates[i][2])
            lines.setThickness(6)
            lnode = lines.create()
            self.linenp = NodePath(lnode)
            self.linenp.instanceTo(self.bbnode)

            #Cloud
            catoms = [atom for atom in chain.get_atoms()]
            for atom in catoms:
                x, y, z = atom.coord
                atomid=atom.get_id()
                a = loader.loadModel("data/atom_sphere")
                a.setPos(x,y,z)
                a.reparentTo(self.nnode)
                a.setColor(ccolor)
                a.setScale(vrad(atomid)*1.1)

        self.bbnode.flattenStrong()
        self.bbnode.hide()
        self.nnode.setTransparency(TransparencyAttrib.MAlpha)
        self.nnode.setAlphaScale(0.3)
        self.nnode.hide()
        
        #DSSP
        self.linenp.instanceTo(self.dsspnode)
        self.struct3 = [dssp[key][2] for key in list(dssp.keys())]    
    
        for i in range(len(self.struct3)):
            dsspcolor = crgbaDSSP(self.struct3[i])
            can_atoms = [atom for atom in self.residues[i] if atom.get_name() == 'CA' or atom.get_name() == 'N']
            for atom in can_atoms:
                x, y, z = atom.coord
                atomid=atom.get_id()
                a = loader.loadModel("data/atom_sphere")
                a.setPos(x, y, z)
                a.reparentTo(self.dsspnode)
                a.setColor(dsspcolor)
                a.setScale(vrad(atomid)/2.5)
            self.dsspnode.flattenStrong()
            self.dsspnode.hide()
        
        #Colocamos la proteina en el centro
        self.cpknode.setPos(0,0,0)
        self.bbnode.setPos(0,0,0)
        self.aanode.setPos(0,0,0)
        self.nnode.setPos(0,0,0)
        
        #Colocamos la camara en el centro
        xc, yc, zc = self.cpknode.getBounds().getCenter()
        self.center = xc, yc, zc
        self.pradius = self.cpknode.getBounds().getRadius()
        self.center_camera()
        
        #Creamos la iluminacion de ambiente
        self.ambient = AmbientLight('alight')
        self.ambient.setColor(LVecBase4f(0.16, 0.16, 0.17, 1.0))
        self.alight = render.attachNewNode(self.ambient)
        render.setLight(self.alight)
        
        #Creamos la iluminacion direccional
        self.directional = DirectionalLight('dlight')
        self.directional.setColor(LVecBase4f(0.8, 0.7, 0.75, 1.0))
        self.directional.setShadowCaster(True,512,512)
        render.setShaderAuto()
        self.dlight = render.attachNewNode(self.directional)
        self.dlight.setPos(0,-50,0)
        render.setLight(self.dlight)
        self.dlight.lookAt(self.cpknode.getBounds().getCenter())
        
        # Post procesado      
        render.setAntialias(AntialiasAttrib.MAuto)
        
        #Teclado
        self.accept('c', self.toggle_cloud)
        self.accept('1', self.showmodel, [self.cpknode])
        self.accept('2', self.showmodel, [self.aanode])
        self.accept('3', self.showmodel, [self.bbnode])
        self.accept('4', self.showmodel, [self.dsspnode])
        self.accept('x', self.center_camera)
        self.accept('arrow_left', self.taskMgr.add, [self.spinCameraTaskX, "SpinCameraTaskX"])
        self.accept('arrow_up', self.taskMgr.add, [self.spinCameraTaskY, "SpinCameraTaskY"])
        self.accept('arrow_down', self.stop_camera)
        self.accept('escape', sys.exit)
        
    def center_camera(self):
        base.cam.setPos(self.center[0], -10-self.center[1]-4*self.pradius, self.center[2])
        base.cam.lookAt(self.center)
        
    def toggle_cloud(self):
        self.cloud = not self.cloud
        if self.cloud:
            self.nnode.show()
        else:
            self.nnode.hide()
            
    def spinCameraTaskX(self,task):
        base.disableMouse()
        taskMgr.remove("SpinCameraTaskY")
        angleDegrees = task.time * 20.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.lookAt(self.center)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont

    def spinCameraTaskY(self,task):
        base.disableMouse()
        taskMgr.remove("SpinCameraTaskX")
        angleDegrees = task.time * 20.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.lookAt(self.center)
        self.camera.setHpr(0, angleDegrees, 0)
        return Task.cont
    
    def stop_camera(self):
        mat=Mat4(camera.getMat())
        mat.invertInPlace()
        base.mouseInterfaceNode.setMat(mat)
        base.enableMouse()
        self.taskMgr.remove("SpinCameraTaskX")
        self.taskMgr.remove("SpinCameraTaskY")
        
    def showmodel(self, node):
        self.cpknode.hide()
        self.aanode.hide()
        self.bbnode.hide()
        self.dsspnode.hide()
        node.show()
            
        
app = PandaViewer()
app.run()