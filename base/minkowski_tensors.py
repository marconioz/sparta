import numpy as np
import numba
import pandas as pd
from numba import jit
import trimesh
import quaternionic
import spherical
import math
#------------helper functions-----------

@jit
def norm(a):
    return np.dot(a,a)**(1/2)


@jit
def fold(num, low, top):

    lower = low
    upper = top

    if num > upper or num == lower:
        num = lower + abs(num + upper) % (abs(lower) + abs(upper))
    if num < lower or num == upper:
        num = upper - abs(num - lower) % (abs(lower) + abs(upper))

    return num

#------------------------------------

class minks:
    def __init__(self,mesh):
        self.mesh = mesh #for debugging purposes
        self.mesh_edges_unique = mesh.edges_unique
        self.ref_vec = np.zeros(3)
        self.area = mesh.area_faces
        self.com = mesh.triangles_center
        self.normal = mesh.face_normals
        self.face_adjacency = mesh.face_adjacency
        self.face_adjacency_edges = mesh.face_adjacency_edges
        self.mesh_vertices = mesh.vertices
        self.mesh_faces = mesh.faces
        self.mesh_triangles = mesh.triangles
        self.face_angles = mesh.face_angles
        self.vertex_faces = mesh.vertex_faces

        self.compute_angles_at_vertice()
        self.compute_faces_edges_length()
        self.compute_adjacent_edge_length()
        self.compute_face_adjacency_matrix()
        self.compute_neighbours_angle()
        self.compute_neighbours_shared_edges_length()
        self.compute_valuations()

    #---------------util main --------------------------

    def compute_faces_edges_length(self):
        #mesh.edges_unique_length[mesh.faces_unique_edges]
        #this computes the length of the sides of all triangles.
        #each line contain the shared edge with another triangle
        #we will never know the label of that triangle though...

        faces_edges = self.mesh.faces_unique_edges
        self.faces_edges_length = self.mesh.edges_unique_length[faces_edges]

    def compute_adjacent_edge_length(self):
        #the edge shared by a two adjacent faces that is...
        #this is not the end of the story...

        fae = self.face_adjacency_edges
        mesh_vertices = self.mesh_vertices
        eal = self.delegate_edge_length(fae,mesh_vertices)
        self.edges_adjacency_length = eal

    def compute_face_adjacency_matrix(self):
        face_adjacency = self.face_adjacency
        dic = {}
        for t1,t2 in face_adjacency:
            dic.setdefault(t1,[]).append(t2)
            dic.setdefault(t2,[]).append(t1)

        #sort of just a little stupid.
        nei_array = np.ones([len(dic),3],int)
        for i in range(len(dic)):
            for j in range(3):
                nei_array[i,j] = int(dic[i][j])

        self.face_adjacency_matrix = nei_array


        # # 1-simplex, 12 neighour faces.
        # facedick = {}
        # for faces in mesh.vertex_faces:
        #     for f1 in faces:
        #         for f2 in faces:
        #             if not f1 == f2:
        #                 if f1 in facedick.keys():
        #                     if f2 not in facedick[f1]:
        #                         facedick[f1].append(f2)
        #                 else:
        #                     facedick[f1] = [f2]
        # # reordering, bit dumb...
        # nt = len(mesh.faces)
        # simplex = np.zeros(shape=[nt,12],dtype=int)#len(facedick.keys()))
        # for i in range(len(facedick.keys())):
        #     simplex[i,:] = [int(j) for j in facedick[i]]

    def compute_neighbours_shared_edges_length(self):

        face_edges = self.mesh.faces_unique_edges #self.face_adjacency_edges
        edges_length = self.mesh.edges_unique_length
        nei_array = self.face_adjacency_matrix
        nei_edges = face_edges[nei_array]
        nt = len(face_edges)

        nei_shared_edges = np.zeros((nt, 3), np.int)
        for l, e in enumerate(face_edges):
            for i in range(3):
                b = nei_edges[l, i, :]
                id = np.argwhere(np.in1d(e, b))
                id = id[0, 0]
                nei_shared_edges[l, i] = e[id]

        self.neighbours_shared_edges_length = edges_length[nei_shared_edges]
        self.neighbours_shared_edges = nei_shared_edges

    def compute_angles_at_vertice(self):
        vertex_faces = self.vertex_faces
        mesh_faces = self.mesh_faces
        face_angles = self.face_angles

        angles_at_vertice = np.zeros(vertex_faces.shape)

        for vertice,faces in enumerate(vertex_faces):
            #sum = sum_of_angles_at_vertice[vertice]
            for fid,face in enumerate(faces):
                if face != -1:
                    vec = mesh_faces[face,:]
                    idx = np.where(vec == vertice)
                    idx = idx[0][0]
                    angles_at_vertice[vertice,fid] = face_angles[face,idx]

                else:
                    angles_at_vertice[vertice,fid] = -1

        self.angles_at_vertice = angles_at_vertice

    def compute_neighbours_angle(self):
        normal = self.normal
        face_adjacency_matrix = self.face_adjacency_matrix
        com = self.com
        alpha = self.delegate_neighbours_angle(normal,face_adjacency_matrix,com)
        self.neighbours_angle = alpha

    def compute_valuations(self):
        scalar = [self.w000,self.w100,self.w200,self.w300]
        vector = [self.w010,self.w110,self.w210,self.w310]
        matrix = [self.w020,self.w120,self.w220,self.w320,self.w102,self.w202]
        tensor = [self.w103,self.w104]
        names_dic = {"scalar": ["w000","w100","w200","w300"],
                     "vector": ["w010","w110","w210","w310"],
                     "matrix": ["w020","w120","w220","w320","w102","w202"],
                     "tensor": ["w103","w104"],
                     "spherical": "sph"}

        self.spherical()

        minkdict = {"scalar": { n:t() for n,t in zip(names_dic['scalar'],scalar)},
                   "vector": { n:t() for n,t in zip(names_dic['vector'],vector)},
                   "matrix": { n:t() for n,t in zip(names_dic['matrix'],matrix)},
                   "tensor": { n:t() for n,t in zip(names_dic['tensor'],tensor)},
                   "spherical": {"sph": self.sphmink}
                   }

        self.minkdict = minkdict


    #----------------util delegated----------------
    @staticmethod
    @jit
    def delegate_edge_length(fae,mesh_vertices):
        edges_length = np.zeros(len(fae))
        for i in fae:
                    v = mesh_vertices[i[0]] - mesh_vertices[i[1]]
                    edges_length[i] = (np.sum(v*v))**(1/2)
        return edges_length

    @staticmethod
    @jit(nopython=True)
    def delegate_neighbours_angle(normal,face_adjacency_matrix,com):
        #this routine is checked. works. not sure it is really necessary though.
        nt = len(normal)
        #no fancy indexing in numba :(
        neighbours_com = np.zeros((nt,3,3))
        neighbours_normal = np.zeros((nt,3,3))
        for t in range(nt):
            for i in range(3):
                for j in range(3):
                    neighbours_com[t,i,j] = com[face_adjacency_matrix[t,i]][j]
                    neighbours_normal[t,i,j] = normal[face_adjacency_matrix[t,i]][j]

        alpha = np.zeros((nt,3))
        for l in range(nt):
            for k in range(3):
                n1 = normal[l]
                nn1 = norm(n1)**(1/2)
                n2 = neighbours_normal[l,k]
                nn2 = norm(n2)**(1/2)

                c = np.dot(n1,n2)/(nn1*nn2)
                # clip not supported yet...or is it?
                if c > 1.0:
                    c = 1
                elif c < -1.0:
                    c = -1

                aux = np.arccos(c)

                convex = com[l] + n1 - (neighbours_com[l,k] + n2)
                concave = com[l] - n1 - (neighbours_com[l,k] - n2);

                if norm(convex) < norm(concave):
                    aux = -aux;
                alpha[l,k] = aux


        return alpha

    #--------------valuations main----------------------------

    #--------------scalar valuations--------------------------

    def w000(self):
        com = self.com
        normal = self.normal
        area = self.area
        w = self.x000(com,normal,area)
        return w

    def w100(self):
        w = np.sum(self.area)/3
        return w

    def w200(self):
        # this is the simpler way of doing this.
        # the way I've done the angle to make sure
        # the edges length matches data matches
        # the angles data. They do.
        # mfa = mesh.face_adjacency_angles
        # fac = mesh.face_adjacency_convex
        # eul = mesh.edges_unique_length
        # flip = [1 if i else -1 for i in fac]
        # np.sum(eul * mfa * flip) / 6
        nsel = self.neighbours_shared_edges_length
        alpha = self.neighbours_angle
        w = self.x200(nsel,alpha)
        return w

    def w300(self):
        w = self.x300(self.angles_at_vertice)
        return w

    #--------------vector valuations--------------------------

    def w010(self):
        w = self.x010(self.mesh_triangles)
        return w

    def w110(self):
        w = self.x110(self.com,self.area)
        return w

    def w210(self):

        w = self.x210(self.mesh_vertices,self.mesh_edges_unique, self.neighbours_angle,self.neighbours_shared_edges_length, self.neighbours_shared_edges)
        return w

    def w310(self):

        w = self.x310(self.angles_at_vertice,self.mesh_vertices)

        return w

    #----------------matrix valuations------------------------
    def w020(self):
        w =  self.x020(self.mesh_triangles,self.normal,self.area,self.ref_vec)
        return w + w.T - np.diag(w.diagonal())

    def w120(self):
        w = self.x120(self.mesh_triangles,self.ref_vec,self.area)
        return w + w.T - np.diag(w.diagonal())

    def w220(self):
        w = self.x220(self.mesh_vertices,self.mesh_edges_unique, self.neighbours_angle,self.neighbours_shared_edges_length, self.neighbours_shared_edges, self.ref_vec)
        return w + w.T - np.diag(w.diagonal())

    def w320(self):
        w = self.x320(self.mesh_vertices,self.angles_at_vertice,self.ref_vec)
        return w + w.T - np.diag(w.diagonal())

    def w102(self):
        w = self.x102(self.normal,self.area)
        return w + w.T - np.diag(w.diagonal())

    def w202(self):
        w = self.x202(self.mesh_vertices,
                      self.mesh_edges_unique,
                      self.normal,
                      self.face_adjacency_matrix,
                      self.neighbours_shared_edges_length,
                      self.neighbours_shared_edges,
                      self.neighbours_angle)

        return w + w.T - np.diag(w.diagonal())


    #--------------tensorial valuations-------------

    def w103(self):
        w = self.x103(self.normal,self.area)
        return w

    def w104(self):
        w = self.x104(self.normal,self.area)
        return w + w.T - np.diag(w.diagonal())


    #--------------spherical valuations------------------


    def spherical(self):
        #todo: correct names from trimesh...
        # check performance of wigner.evaluate/rotate
        area = self.mesh.area_faces
        total_area = area.sum()
        weighted_area = area/total_area
        normal = self.mesh.face_normals
        nt = len(normal)
        l_max = 8
        sphmink = np.zeros(l_max)
        wigner = spherical.Wigner(l_max)
        for l in range(0,l_max):
            prefactor = 4*np.pi/(2*l + 1)
            for t in range(nt):
                [x,y,z] = normal[t]
                theta = np.arccos(z)
                phi = np.arctan2(y, x)
                R = quaternionic.array.from_spherical_coordinates(theta, phi)
                Y = wigner.sYlm(0, R)
                for m in range(-l,l):
                    sphmink[l] += np.abs(prefactor*weighted_area[t]*Y[wigner.Yindex(l, m)])

        self.sphmink = sphmink


    #--------------valuations delegated-----------------------

    #--------------scalar valuations  [checked, all good!] --------------------------

    @staticmethod
    @jit(nopython=True)
    def x000(com,normal,area):
        w = 0
        for i in range(len(com)):
            a = np.dot(com[i],normal[i])*area[i]
            w += a*(1./3)
        return w

    @staticmethod
    @jit
    def x200(neighbours_shared_edges_length,neighbours_angle):

        w = np.sum(neighbours_shared_edges_length*neighbours_angle)
        return w/12

    @staticmethod
    @jit
    def x300(angles_at_vertice):
        #beware of bad triangles in meshes.
        #sum of angles must be larger than zero.
        av = angles_at_vertice
        bux = 0
        for v in av:
            aux = 0
            for a in v[v > 0]:
                    aux += a
            if aux >0 :
                bux += 2*np.pi - aux

        return bux/3



    #---------------vector valuations  [checked, all good!] --------------

    @staticmethod
    @jit
    def x010(mesh_triangles):
        #center of mass * volume
        nt = len(mesh_triangles)
        ww = np.zeros(3)
        for i in range(3):
            for j in range(nt):
                c1 = mesh_triangles[j,0]
                c2 = mesh_triangles[j,1]
                c3 = mesh_triangles[j,2]
                v = c2-c1
                w = c3-c1
                part1 = 2*v[i]*v[(i+1)%3] \
                        + v[(i+1)%3]*w[i] \
                        + 4*c1[(i+1)%3]*(v[i]+w[i]) \
                        + v[i]*w[(i+1)%3] \
                        + 2*w[i]*w[(i+1)%3] \
                        + 4*c1[i]*(3*c1[(i+1)%3]+v[(i+1)%3]+w[(i+1)%3])
                vf = np.cross(v,w)[(i+1)%3]
                ww[i] += vf*part1/24
        return ww


    @staticmethod
    @jit
    def x110(com,area):
        nt = len(com)
        w = np.zeros(3)
        for i in range(3):
            for j in range(nt):
                part = com[j,i]*area[j]
                w[i] += part/3.0
        return w
    @staticmethod
    @jit
    def x210(vertices,edges_unique,neighbours_angle,neighbours_shared_edges_length,neighbours_shared_edges):
        alpha =  neighbours_angle
        nsel = neighbours_shared_edges_length
        nse =  neighbours_shared_edges
        w = np.zeros(3)
        for i, n in enumerate(nse):
            for j, e in enumerate(n):
                c1 = vertices[edges_unique[e]][0]
                c2 = vertices[edges_unique[e]][1]
                c12 = c1 + c2
                w += alpha[i, j] * nsel[i, j] * c12
        return w/24

        # the other way:
        # mesh = mm.mesh
        # alpha = mesh.face_adjacency_angles
        # fac = mesh.face_adjacency_convex
        # el = mesh.edges_unique_length
        # flip = [1 if i else -1 for i in fac]
        # alpha = alpha * flip
        # c = mm.mesh_vertices[mm.face_adjacency_edges]
        # c12 = c[:, 1, :] + c[:, 0, :]
        # part = alpha * el
        # w = np.einsum('i,ij', part, c12) / 12



    @staticmethod
    @jit
    def x310(angles_at_vertice,mesh_vertices):
        angles = angles_at_vertice
        mv = mesh_vertices
        angles_sum = [np.sum(v[v > 0]) for v in angles]
        nv = len(mesh_vertices)
        w = np.zeros(3)
        for i in range(3):
            for j in range(nv):
                for k in range(len(angles[j])):
                    if angles[j, k] > 0:
                        w_part = ((2 * np.pi * (angles[j, k] / angles_sum[j])) - angles[j, k]) * mv[j, i]
                        w[i] += w_part / 3.
        return w

        # old and wrong
        # angles = angles_at_vertice
        # angles_sum = angles.sum(axis=1)
        # nv = len(mesh_vertices)
        # w = np.zeros(3)
        # for i in range(3):
        #     for j in range(nv):
        #         for k in range(len(angles[j])):
        #             w_part = ((2*np.pi*(angles[j,k]/angles_sum[j])) - angles[j,k])*mesh_vertices[j,i];
        #             w[i] += w_part/3.


    #----------------matrix valuations [checked, all good!] ------------------------
    # checked: good
    @staticmethod
    @jit
    def x020(mesh_triangles,normal,area,ref_vec):
        w = np.zeros((3,3))
        mt = mesh_triangles
        nt = len(mt)
        for k in range(nt):
            c1 = mt[k,0] - ref_vec
            c2 = mt[k,1] - ref_vec
            c3 = mt[k,2] - ref_vec
            n = normal[k]

            Ixx = (3*pow(c2[0],2)*c2[2] + 2*c2[0]*c2[2]*c3[0] + c2[2]*pow(c3[0],2)
                   + c1[2]*(pow(c2[0],2) + c2[0]*c3[0] + pow(c3[0],2))
                   + pow(c2[0],2)*c3[2] + 2*c2[0]*c3[0]*c3[2] + 3*pow(c3[0],2)*c3[2]
                   + pow(c1[0],2)*(3*c1[2] + c2[2] + c3[2])
                   + c1[0]*(2*c1[2]*(c2[0] + c3[0]) + c2[0]*(2*c2[2] + c3[2])
                            + c3[0]*(c2[2] + 2*c3[2])))/60.
            w[0,0] += Ixx*2*area[k]*n[2]

            Iyy = (3*pow(c2[1],2)*c2[2] + 2*c2[1]*c2[2]*c3[1] + c2[2]*pow(c3[1],2)
                   + c1[2]*(pow(c2[1],2) + c2[1]*c3[1] + pow(c3[1],2))
                   + pow(c2[1],2)*c3[2] + 2*c2[1]*c3[1]*c3[2] + 3*pow(c3[1],2)*c3[2]
                   + pow(c1[1],2)*(3*c1[2] + c2[2] + c3[2])
                   + c1[1]*(2*c1[2]*(c2[1] + c3[1]) + c2[1]*(2*c2[2] + c3[2])
                            + c3[1]*(c2[2] + 2*c3[2])))/60.
            w[1,1] += Iyy*2*area[k]*n[2]

            Izz = (3*c2[1]*pow(c2[2],2) + pow(c2[2],2)*c3[1]
                   + pow(c1[2],2)*(c2[1] + c3[1]) + 2*c2[1]*c2[2]*c3[2]
                   + 2*c2[2]*c3[1]*c3[2] + c2[1]*pow(c3[2],2) + 3*c3[1]*pow(c3[2],2)
                   + c1[1]*(3*pow(c1[2],2) + pow(c2[2],2) + c2[2]*c3[2]
                            + pow(c3[2],2) + 2*c1[2]*(c2[2] + c3[2]))
                   + c1[2]*(c2[1]*(2*c2[2] + c3[2]) + c3[1]*(c2[2] + 2*c3[2])))/60.
            w[2,2] += Izz*2*area[k]*n[1]

            Ixy = (2*c1[2]*c2[0]*c2[1] + 6*c2[0]*c2[1]*c2[2] + c1[2]*c2[1]*c3[0]
                   + 2*c2[1]*c2[2]*c3[0] + c1[2]*c2[0]*c3[1] + 2*c2[0]*c2[2]*c3[1]
                   + 2*c1[2]*c3[0]*c3[1] + 2*c2[2]*c3[0]*c3[1] + 2*c2[0]*c2[1]*c3[2]
                   + 2*c2[1]*c3[0]*c3[2] + 2*c2[0]*c3[1]*c3[2] + 6*c3[0]*c3[1]*c3[2]
                   + c1[0]*(2*c2[1]*c2[2] + c2[2]*c3[1] + 2*c1[2]*(c2[1] + c3[1])
                            + c2[1]*c3[2] + 2*c3[1]*c3[2]
                            + 2*c1[1]*(3*c1[2] + c2[2] + c3[2]))
                   + c1[1]*(2*c1[2]*(c2[0] + c3[0]) + c2[0]*(2*c2[2] + c3[2])
                            + c3[0]*(c2[2] + 2*c3[2])))/120.
            w[0,1] += Ixy*2*area[k]*n[2]
            w[0,2] += Ixy*2*area[k]*n[1]
            w[1,2] += Ixy*2*area[k]*n[0]

        return w


    #checked: good
    @staticmethod
    @jit
    def x120(mesh_triangles,ref_vec,area):
        mt = mesh_triangles
        nt = len(mt)
        w = np.zeros((3,3))
        for k in range(nt):
            c1 = mt[k,0] - ref_vec
            c2 = mt[k,1] - ref_vec
            c3 = mt[k,2] - ref_vec

            for i in range(3):
                for j in range(i,3):
                    part_1 = c1[i]*c1[j]+c2[i]*c2[j]+c3[i]*c3[j]
                    part_2 = c1[i]*c2[j]+c2[i]*c3[j]+c3[i]*c1[j]
                    part_3 = c1[j]*c2[i]+c2[j]*c3[i]+c3[j]*c1[i]
                    w[i,j] += 1/18.*(part_1+part_2/2.+part_3/2.)*area[k]

        return w

    # checked: good
    @staticmethod
    @jit
    def x220(vertices,edges_unique,neighbours_angle,neighbours_shared_edges_length,neighbours_shared_edges,ref_vec):
        alpha = neighbours_angle
        nsel = neighbours_shared_edges_length
        nse = neighbours_shared_edges
        w = np.zeros((3,3))
        for k, n in enumerate(nse):
            for l, e in enumerate(n):
                c1 = vertices[edges_unique[e]][0] - ref_vec
                c2 = vertices[edges_unique[e]][1] - ref_vec
                for i in range(3):
                    for j in range(i, 3):
                        c12 = c1[i]*c1[j] + 0.5*(c1[i]*c2[j] + c1[j]*c2[i]) + c2[i] * c2[j]
                        w[i, j] += alpha[k,l]*nsel[k,l]*c12/(18.*2.)

        return w


    # checked: good
    @staticmethod
    @jit
    def x320(mesh_vertices,angles_at_vertice,ref_vec):

        sum_of_angles_at_vertice = [np.sum(v[v > 0]) for v in angles_at_vertice]

        w = np.zeros((3,3))

        for j in range(len(mesh_vertices)):
            c = mesh_vertices[j] - ref_vec
            for k in range(len(angles_at_vertice[j])):
                if angles_at_vertice[j, k] > 0:
                    rate = angles_at_vertice[j,k]/sum_of_angles_at_vertice[j]
                    angle_part = 2*np.pi*rate - angles_at_vertice[j,k]

                    for m in range(3):
                        for l in range(m,3):
                            w_part = angle_part*c[m]*c[l];
                            w[m,l] += w_part/3.

        return w


    # checked: good
    @staticmethod
    @jit
    def x102(normal,area):
        w = np.zeros((3,3))
        nt = len(normal)
        for k in range(nt):
            n = normal[k]
            for i in range(3):
                for j in range(i,3):
                    w[i,j] += 1/3.*area[k]*n[i]*n[j];
        return w

    # checked: good
    @staticmethod
    @jit
    def x202(vertices,
             edges_unique,
             normal,
             face_adjacency_matrix,
             neighbours_shared_edges_length,
             neighbours_shared_edges,
             neighbours_angle):

        nt = len(normal)

        # move this to property?
        neighbours_normal = np.zeros((nt,3,3))
        for t in range(nt):
            for i in range(3):
                for j in range(3):
                    neighbours_normal[t,i,j] = normal[face_adjacency_matrix[t,i]][j]

        w = np.zeros((3,3))
        for k, n in enumerate(neighbours_shared_edges):
            for l, e in enumerate(n):
                c1 = vertices[edges_unique[e]][0]
                c2 = vertices[edges_unique[e]][1]

                alpha =  neighbours_angle[k,l]
                n1 = normal[k]
                n2 = neighbours_normal[k,l]
                e_norm = neighbours_shared_edges_length[k,l]

                e = (c2 - c1)/e_norm
                n_a = (n1 + n2)/(norm(n1+n2))
                n_i = np.cross(e,n_a)

                for i in range(3):
                    for j in range(i,3):
                        local_value = e_norm*((alpha + np.sin(alpha))*n_a[i]*n_a[j]+(alpha - np.sin(alpha))*n_i[i]*n_i[j])
                        w[i,j] += 1/12.*1/2.*local_value;
        return w

    # -------------------------tensorial valuations--------


    # checked: good
    @staticmethod
    @jit
    def x103(normal,area):
        nt = len(area)
        w = np.zeros((3,3,3))
        for l in range(nt):
            a = area[l]
            n = normal[l]
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        w[i,j,k] += (1/3.)*a*n[i]*n[j]*n[k]
        return w


    # checked:
    # bad: does not match whatever karambola produces.
    # additivity?
    # invariance?

    @staticmethod
    @jit
    def x104(normal,area):
        nt = len(area)
        w = np.zeros((6,6))
        for k in range(nt):
            a = area[k]
            n = normal[k]
            x = n[0]
            y = n[1]
            z = n[2]
            s2 = 2**(0.5)
            t = [x*x,y*y,z*z,y*z*s2,x*z*s2,x*y*s2]

            for i in range(6):
                for j in range(0,i):
                    ftp = t[i]*t[j] # fourth tensorial power
                    w[i,j] += a*ftp/3.


        return w

#%%