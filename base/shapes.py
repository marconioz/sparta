import trimesh
import pyvista
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display,HTML
from base.minkowski_tensors import *

jnames = {'J1': 'square pyramid ',
          'J2': 'pentagonal pyramid ',
          'J3': 'triangular cupola ',
          'J4': 'square cupola ',
          'J5': 'pentagonal cupola ',
          'J6': 'pentagonal rotunda ',
          'J7': 'elongated triangular pyramid ',
          'J8': 'elongated square pyramid ',
          'J9': 'elongated pentagonal pyramid ',
          'J10': 'gyroelongated square pyramid ',
          'J11': 'gyroelongated pentagonal pyramid ',
          'J12': 'triangular dipyramid ',
          'J13': 'pentagonal dipyramid ',
          'J14': 'elongated triangular dipyramid ',
          'J15': 'elongated square dipyramid ',
          'J16': 'elongated pentagonal dipyramid ',
          'J17': 'gyroelongated square dipyramid ',
          'J18': 'elongated triangular cupola ',
          'J19': 'elongated square cupola ',
          'J20': 'elongated pentagonal cupola ',
          'J21': 'elongated pentagonal rotunda ',
          'J22': 'gyroelongated triangular cupola ',
          'J23': 'gyroelongated square cupola ',
          'J24': 'gyroelongated pentagonal cupola ',
          'J25': 'gyroelongated pentagonal rotunda ',
          'J26': 'gyrobifastigium ',
          'J27': 'triangular orthobicupola ',
          'J28': 'square orthobicupola ',
          'J29': 'square gyrobicupola ',
          'J30': 'pentagonal orthobicupola ',
          'J31': 'pentagonal gyrobicupola ',
          'J32': 'pentagonal orthocupolarontunda ',
          'J33': 'pentagonal gyrocupolarotunda ',
          'J34': 'pentagonal orthobirotunda ',
          'J35': 'elongated triangular orthobicupola ',
          'J36': 'elongated triangular gyrobicupola ',
          'J37': 'elongated square gyrobicupola ',
          'J38': 'elongated pentagonal orthobicupola ',
          'J39': 'elongated pentagonal gyrobicupola ',
          'J40': 'elongated pentagonal orthocupolarotunda ',
          'J41': 'elongated pentagonal gyrocupolarotunda ',
          'J42': 'elongated pentagonal orthobirotunda ',
          'J43': 'elongated pentagonal gyrobirotunda ',
          'J44': 'gyroelongated triangular bicupola ',
          'J45': 'gyroelongated square bicupola ',
          'J46': 'gyroelongated pentagonal bicupola ',
          'J47': 'gyroelongated pentagonal cupolarotunda ',
          'J48': 'gyroelongated pentagonal birotunda ',
          'J49': 'augmented triangular prism ',
          'J50': 'biaugmented triangular prism ',
          'J51': 'triaugmented triangular prism ',
          'J52': 'augmented pentagonal prism ',
          'J53': 'biaugmented pentagonal prism ',
          'J54': 'augmented hexagonal prism ',
          'J55': 'parabiaugmented hexagonal prism ',
          'J56': 'metabiaugmented hexagonal prism ',
          'J57': 'triaugmented hexagonal prism ',
          'J58': 'augmented dodecahedron ',
          'J59': 'parabiaugmented dodecahedron ',
          'J60': 'metabiaugmented dodecahedron ',
          'J61': 'triaugmented dodecahedron ',
          'J62': 'metabidiminished icosahedron ',
          'J63': 'tridiminished icosahedron ',
          'J64': 'augmented tridiminished icosahedron ',
          'J65': 'augmented truncated tetrahedron ',
          'J66': 'augmented truncated cube ',
          'J67': 'biaugmented truncated cube ',
          'J68': 'augmented truncated dodecahedron ',
          'J69': 'parabiaugmented truncated dodecahedron ',
          'J70': 'metabiaugmented truncated dodecahedron ',
          'J71': 'triaugmented truncated dodecahedron ',
          'J72': 'gyrate rhombicosidodecahedron ',
          'J73': 'parabigyrate rhombicosidodecahedron ',
          'J74': 'metabigyrate rhombicosidodecahedron ',
          'J75': 'trigyrate rhombicosidodecahedron ',
          'J76': 'diminished rhombicosidodecahedron ',
          'J77': 'paragyrate diminished rhombicosidodecahedron ',
          'J78': 'metagyrate diminished rhombicosidodecahedron ',
          'J79': 'bigyrate diminished rhombicosidodecahedron ',
          'J80': 'parabidiminished rhombicosidodecahedron ',
          'J81': 'metabidiminished rhombicosidodecahedron ',
          'J82': 'gyrate bidiminished rhombicosidodecahedron ',
          'J83': 'tridiminished rhombicosidodecahedron ',
          'J84': 'snub disphenoid ',
          'J85': 'snub square antiprism ',
          'J86': 'sphenocorona ',
          'J87': 'augmented sphenocorona ',
          'J88': 'sphenomegacorona ',
          'J89': 'hebesphenomegacorona ',
          'J90': 'disphenocingulum ',
          'J91': 'bilunabirotunda ',
          'J92': 'triangular hebesphenorotunda '}

def flatten_minks(compo):
    mf = []
    dicks = {}
    for name, mesh in compo.items():
        mdick = minks(mesh).minkdic
        print(mesh.volume)
        mink_features = {}
        for key, value in mdick.items():
            if key == 'scalar':
                dick = {n: value[n] for n in value}
                mink_features[key] = {n: value[n] for n in value}
            else:
                aux = {}
                for n in value:
                    for p, k in enumerate(np.ravel(value[n])):
                        features = np.ravel(value[n])
                        aux[n + "_" + str(p)] = k
                        dick[n + "_" + str(p)] = k
                mink_features[key] = aux
        dicks[name] = dick
        mf.append(mink_features)
    return (dicks, mf)
def triangulate(face):
    triangles = []
    for i in range(1, len(face) - 1):
        triangles.append([face[0], face[i], face[i + 1]])
    return (triangles)
class Johnson_Polyhedra:
    library = {}

    def __init__(self, frontend):
        self.frontend = frontend  # pyvista, trimesh, plotly
        self.jays()

    def show(self, ids):
        what = self.library  # is a pandas pd.
        toplot = []
        names = []
        for id in ids:
            toplot.append(what.loc[id].to_dict())
        if self.frontend == 'plotly':
            '''plotly front-end do most visuals as the others, and is more stable, and gives us less pain to install/manage.'''
            '''pyvista (panel is a pain) and trimesh is less pain'''
            fig = go.Figure()

            for id, w in enumerate(toplot):

                points = 15 * np.random.rand(1, 3) + np.array(w['vertices'])
                faces = w['faces']
                new_faces = []
                for f in faces:
                    if len(f) > 3:
                        triangles = triangulate(f)
                        for t in triangles:
                            new_faces.append(t)
                    else:
                        new_faces.append(f)

                x, y, z = np.array(points).T
                i, j, k = np.array(new_faces).T

                trace = go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    name=w['long_name'],
                    showscale=True,
                )
                fig.add_trace(trace)

            fig.update_layout(scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)),
                width  = 700,
                height = 700,
                margin=dict(r=0, l=0,b=0, t=0)
            )

            fig.update_traces(
                text=ids,
                hovertemplate="<br>".join([
                    "Solid Name: %{text}"])
            )
            return fig
            # fig.show()
        if self.frontend == 'pyvista':
            p = pyvista.Plotter(notebook=True)
            for w in toplot:
                surf = pyvista.PolyData()
                surf.points = 15 * np.random.rand(1, 3) + np.array(w['vertices'])
                # typical kitware complication, faces are encoded in a funny way:
                bux = []

                for f in w['faces']:
                    aux = [len(f)]
                    for e in f:
                        aux.append(e)
                    bux.append(aux)

                surf.faces = np.hstack(bux)
                p.add_mesh(surf, scalars=np.arange(len(bux)))
                # surf.plot(scalars=np.arange(len(bux)), cpos=[-1, 1, 0.5],jupyter_backend='panel')
            # p.show_grid(color='black')
            # p.set_background(color='white')
            p.show(jupyter_backend='panel')
        if self.frontend == 'trimesh':
            import trimesh
            meshes = []
            for w in toplot:
                points = 15 * np.random.rand(1, 3) + np.array(w['vertices'])
                faces = w['faces']
                new_faces = []
                for f in faces:
                    if len(f) > 3:
                        triangles = triangulate(f)
                        for t in triangles:
                            new_faces.append(t)
                    else:
                        new_faces.append(f)

                body_mesh = trimesh.Trimesh(vertices=points,
                                            faces=new_faces,
                                            process=True)
                body_color = trimesh.visual.random_color()

                body_mesh.visual.face_colors = body_color
                meshes.append(body_mesh)

            self.meshes = meshes

            # a more absurd hack!
            sh = np.sum(meshes).show()
            HEX_CODE = '000000'
            shd = sh.data.replace('scene.background=new THREE.Color(0xffffff)', f'scene.background=new THREE.Color(0x{HEX_CODE})')
            display(HTML(shd))  # ,width=300,height=300))

    def grouping(self):
        # platonics
        # aristotelian
        # snubs...
        pass

    def jays(self):
        rdf = pd.read_pickle('../data/johnsons_pandas_df')
        rdf = rdf.set_index('short_name')
        self.library = rdf