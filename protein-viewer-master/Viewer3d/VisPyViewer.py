# Code based on Luke Campagnola's Spheres visual, 2018.

import numpy as np
from vispy import app, gloo, visuals, scene

vertex = """
#version 120

uniform vec3 u_light_position;
uniform vec3 u_light_spec_position;

attribute vec3  a_position;
attribute vec3  a_color;
attribute float a_radius;

varying vec3  v_color;
varying vec4  v_eye_direction;
varying float v_radius;
varying vec3  v_light_direction;

varying float v_depth;
varying float v_depth_radius;

void main (void) {
    vec4 atom_pos = vec4(a_position, 1);
    
    // First decide where to draw this atom on screen
    vec4 fb_pos = $visual_to_framebuffer(atom_pos);
    gl_Position = $framebuffer_to_render(fb_pos);
    
    // Measure the orientation of the framebuffer coordinate system relative
    // to the atom
    vec4 x = $framebuffer_to_visual(fb_pos + vec4(100, 0, 0, 0));
    x = (x/x.w - atom_pos) / 100;
    vec4 z = $framebuffer_to_visual(fb_pos + vec4(0, 0, -100, 0));
    z = (z/z.w - atom_pos) / 100;
    
    // Use the x axis to measure radius in framebuffer pixels
    // (gl_PointSize uses the framebuffer coordinate system)
    vec4 radius = $visual_to_framebuffer(atom_pos + normalize(x) * a_radius);
    radius = radius/radius.w - fb_pos/fb_pos.w;
    gl_PointSize = length(radius);
    
    // Use the z axis to measure position and radius in the depth buffer
    v_depth = gl_Position.z / gl_Position.w;
    // gl_FragDepth uses the "render" coordinate system.
    vec4 depth_z = $framebuffer_to_render($visual_to_framebuffer(atom_pos + normalize(z) * a_radius));
    v_depth_radius = v_depth - depth_z.z / depth_z.w;
    
    v_light_direction = normalize(u_light_position);
    v_radius = a_radius;
    v_color = a_color;
}
"""

fragment = """
#version 120

varying vec3  v_color;
varying float v_radius;
varying vec3  v_light_direction;
varying float v_depth;
varying float v_depth_radius;

void main()
{
    // calculate xyz position of this fragment relative to radius
    vec2 texcoord = gl_PointCoord * 2.0 - vec2(1.0);
    float x = texcoord.x;
    float y = texcoord.y;
    float d = 1.0 - x*x - y*y;
    if (d <= 0.0)
        discard;
    float z = sqrt(d);
    vec3 normal = vec3(x,y,z);
    
    // Diffuse color
    float ambient = 0.3;
    float diffuse = dot(v_light_direction, normal);
    // clamp, because 0 < theta < pi/2
    diffuse = clamp(diffuse, 0.0, 1.0);
    vec3 light_color = vec3(1, 1, 1);
    vec3 diffuse_color = ambient + light_color * diffuse;

    // Specular color
    //   reflect light wrt normal for the reflected ray, then
    //   find the angle made with the eye
    vec3 eye = vec3(0, 0, -1);
    float specular = dot(reflect(v_light_direction, normal), eye);
    specular = clamp(specular, 0.0, 1.0);
    // raise to the material's shininess, multiply with a
    // small factor for spread
    specular = pow(specular, 80);
    vec3 specular_color = light_color * specular;
    
    gl_FragColor = vec4(v_color * diffuse_color + specular_color, 1);
    gl_FragDepth = v_depth - .5 * z * v_depth_radius;
}
"""

class SpheresVisual(visuals.Visual):
    """Visual that draws many spheres.
    
    Parameters
    ----------
    coordinates: array of coordinates
    color: array of colors
    radius: array of radius
    """
    def __init__(self, coordinates, color, radius):
        visuals.Visual.__init__(self, vertex, fragment)
        
        self.natoms = len(coordinates)
        
        #Loading data and type
        self._load_data()
        self._draw_mode = 'points'
        self.set_gl_state('translucent', depth_test=True, cull_face=False)        
        
    def _load_data(self):
        """Make an array with all the data and load it into VisPy Gloo"""
        data = np.zeros(self.natoms, [('a_position', np.float32, 3),
                            ('a_color', np.float32, 4),
                            ('a_radius', np.float32, 1)])

        data['a_position'] = coordinates
        data['a_color'] = color
        data['a_radius'] = radius#*view.transforms.pixel_scale

        self.shared_program.bind(gloo.VertexBuffer(data))
        
        self.shared_program['u_light_position'] = 5., -5., 5.
    
    def _prepare_transforms(self,view):
        view.view_program.vert['visual_to_framebuffer'] = view.get_transform('visual', 'framebuffer')
        view.view_program.vert['framebuffer_to_visual'] = view.get_transform('framebuffer', 'visual')
        view.view_program.vert['framebuffer_to_render'] = view.get_transform('framebuffer', 'render')
        
        
from Bio.PDB import PDBParser,DSSP
# from moleculardata import crgbaDSSP, restype, colorrgba, vrad, resdict
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
def centroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x/length, sum_y/length, sum_z/length

def atom_information(pdbdata,mode):
    
    #analyze pdb file
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure =parser.get_structure('model',pdbdata)
    
    #DSSP prediction
    pmodel = structure[0]
    dssp = DSSP(pmodel,pdbdata)
    
    #Set variables
    global coordinates
    global color
    global radius
    global chains
    global chain_coords
    global chain_colors
    
    if mode == 'cpk':
        #list of atoms
        atoms = [atom for atom in structure.get_atoms()]
        natoms = len(atoms)
        #atom coordinates
        coordinates = np.array([atom.coord for atom in atoms])
        center = centroid(coordinates)
        coordinates -= center

        #atom color
        color = [colorrgba(atom.get_id()) for atom in atoms]
        #atom radius
        radius = np.array([vrad(atom.get_id()) for atom in atoms])
        
    elif mode == 'aminoacid':
        #list of atoms
        atoms = [atom for atom in structure.get_atoms() if atom.get_parent().resname != 'HOH']
        natoms = len(atoms)
        #atom coordinates
        coordinates = np.array([atom.coord for atom in atoms])
        center = centroid(coordinates)
        coordinates -= center
        #atom color
        color = [colorrgba(restype(atom.get_parent().resname)) for atom in atoms]
        #atom radius
        radius = np.array([vrad(atom.get_id()) for atom in atoms])
        
    elif mode == 'backbone':
        #list of atoms
        atoms = [atom for atom in structure.get_atoms() if atom.get_name() == 'CA' or atom.get_name() == 'N']
        natoms = len(atoms)
        #atom coordinates
        coordinates = np.array([atom.coord for atom in atoms])
        center = centroid(coordinates)
        coordinates -= center
        #atom color
        color = []
        #list of arrays of coordinates and colors for each chain
        chains = []
        chain_colors = []
        chain_coords=[]
        for chain in structure.get_chains():
            chains.append(chain)
            can_coord = np.array([atom.coord for atom in chain.get_atoms() if atom.get_name() =='CA' or atom.get_name() =='N'])
            can_coord -= center
            chain_coords.append(can_coord)
            chain_length = len(can_coord)
            chain_color = np.append(np.random.rand(1,3),[1.0])
            chain_colors.append(chain_color)
            color.append(np.tile(chain_color,(chain_length,1)))
        if len(chains)>1:
            color = np.concatenate(color)
        #atom radius
        radius = np.array([vrad(atom.get_id()) for atom in atoms])
    
    elif mode == 'dssp':
        #list of atoms
        atoms = [atom for atom in structure.get_atoms() if atom.get_name() == 'CA' or atom.get_name() == 'N']
        natoms = len(atoms)
        #atom coordinates
        coordinates = np.array([atom.coord for atom in atoms])
        center = centroid(coordinates)
        coordinates -= center
        #atom color
        struct3 = [dssp[key][2] for key in list(dssp.keys())]
        residues = [residue for residue in structure.get_residues() if residue.get_resname() in resdict.keys()]
        color = []
        for i in range(len(struct3)):
            dsspcolor = crgbaDSSP(struct3[i])
            n_atoms = len([atom for atom in residues[i] if atom.get_name() =='CA' or atom.get_name() == 'N'])
            color.append(np.tile(dsspcolor,(n_atoms,1)))
        if len(struct3)>1:
            color = np.concatenate(color)
        #list of arrays of coordinates and colors for each chain
        chains = []
        chain_colors = []
        chain_coords =[]
        for chain in structure.get_chains():
            chains.append(chain)
            chain_color = np.append(np.random.rand(1,3),[1.0])
            chain_colors.append(chain_color) 
            can_coord = np.array([atom.coord for atom in chain.get_atoms() if atom.get_name() =='CA' or atom.get_name() =='N'])
            can_coord -= center
            chain_coords.append(can_coord)
        #atom radius
        radius = np.array([vrad(atom.get_id()) for atom in atoms])

class VisPyViewer(object):
    visualization_modes = ['cpk','backbone','aminoacid', 'dssp']
    
    def __init__(self, pdbdata, mode='cpk'):

        #Mode selection
        if mode not in VisPyViewer.visualization_modes:
            raise Exception('Not recognized visualization mode %s' % mode)
        self.mode = mode
        
        #Data selection
        atom_information(pdbdata, self.mode)

        self.radius = max(abs(np.concatenate(coordinates)))
        
        #Canvas + camera
        canvas = scene.SceneCanvas(keys='interactive', app='pyqt5', bgcolor='white', size=(1200,800), show=True)
        view = canvas.central_widget.add_view()
        view.camera = scene.ArcballCamera(fov=70, distance=(self.radius+40))
        
        #Load visual and apply it
        Spheres = scene.visuals.create_visual_node(SpheresVisual)
        Lines = scene.visuals.create_visual_node(visuals.LinePlotVisual)
        vis_atoms=[Spheres(coordinates,color,radius,parent=view.scene)]
        vis_chains=[]
        if self.mode in ['backbone','dssp']:
            for i in range(len(chains)):
                vis_chains.append(Lines(chain_coords[i], color = chain_colors[i],parent=view.scene))

        #Run the program
        canvas.app.run()
