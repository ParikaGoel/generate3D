import os
import math
import plyfile
import argparse
import numpy as np
import tinyobjloader
from PIL import Image


def write_off(file, vertices, faces):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert len(face) == 3, 'only triangular faces supported (%s)' % file

            fp.write("%d %d %d %d\n" % (len(face), face[0], face[1], face[2]))

        # add empty line to be sure
        fp.write('\n')


def write_coff(file, vertices, faces):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z, r, g, b) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('COFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            fp.write(
                str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + ' ' + str(int(vertex[3])) + ' ' + str(
                    int(vertex[4])) + ' ' + str(int(vertex[5])) + '\n')

        for face in faces:
            assert len(face) == 3, 'only triangular faces supported (%s)' % file

            fp.write("%d %d %d %d\n" % (len(face), face[0], face[1], face[2]))

        # add empty line to be sure
        fp.write('\n')


def write_ply(file, vertices, faces):
    """
        Writes the given vertices and faces to OFF.

        :param vertices: vertices as tuples of (x, y, z, r, g, b) coordinates
        :type vertices: [(float)]
        :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
        :type faces: [(int)]
        """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            element face %d
            property list uchar int vertex_index
            end_header
            ''' % (num_vertices, num_faces))

        for vert in vertices:
            if len(vert) == 3:
                color = (0, 169, 255)
                fp.write('3 %f %f %f %d %d %d\n' % vert+color)
            elif len(vert) == 6:
                fp.write('3 %f %f %f %d %d %d\n' % vert)
            else:
                print('Error: Incorrect number of properties in a vertex. Expected 3 or 6 entries\n')
                return

        for face in faces:
            fp.write('3 %d %d %d\n' % face)

        # add empty line to be sure
        fp.write('\n')


def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        assert lines[0] == 'OFF' or lines[0] == 'off', 'invalid OFF file %s' % file

        parts = lines[1].split(' ')
        assert len(parts) == 3

        num_vertices = int(parts[0])
        assert num_vertices > 0

        num_faces = int(parts[1])
        assert num_faces > 0

        start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, 'face should have %d vertices but as %d (%s)' % (
                face[0], len(face) - 1, file)
            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (
                    index, num_vertices, file)

            assert len(face) > 1

            faces.append(face)

        return vertices, faces


def read_ply(filename):
    """
    Reads vertices and faces from a ply file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """
    plydata = plyfile.PlyData().read(filename)

    vertices = [list(x) for x in plydata['vertex']]  # Element 0 is the vertex
    faces = [x.tolist() for x in plydata['face'].data['vertex_indices']]  # Element 1 is the face
    return vertices, faces


def read_obj(filename):
    reader = tinyobjloader.ObjReader()
    ret = reader.ParseFromFile(filename)

    if ret == False:
        print("Warn:", reader.Warning())
        print("Err:", reader.Error())
        print("Failed to load : ", filename)

        sys.exit(-1)

    attrib = reader.GetAttrib()
    shapes = reader.GetShapes()

    faces = []

    vertices = list(map(tuple, np.asarray(attrib.vertices).reshape(-1,3)))

    for shape in shapes:
        index_offset = 0

        for fv in shape.mesh.num_face_vertices:
            face = []
            for v in range(fv):
                face.append(shape.mesh.indices[index_offset+v].vertex_index)
            faces.append(face)
            index_offset = index_offset + fv

    return vertices, faces

if __name__ == '__main__':
    obj_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-data/04379243/1abfb0c03c81fc2219fb4103277a6b93/models/model_normalized.obj"
    off_file = "/home/parika/WorkingDir/complete3D/Assets/raw/model.off"

    vertices, faces = read_obj(obj_file)
    write_off(off_file, vertices, faces)