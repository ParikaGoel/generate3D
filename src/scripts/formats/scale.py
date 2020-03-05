import numpy as np
import data_formats as formatHandler

class Mesh:
    """
    Represents a mesh.
    """

    def __init__(self, vertices=[[]], faces=[[]]):
        """
        Construct a mesh from vertices and faces.

        :param vertices: list of vertices, or numpy array
        :type vertices: [[float]] or numpy.ndarray
        :param faces: list of faces or numpy array, i.e. the indices of the corresponding vertices per triangular face
        :type faces: [[int]] or numpy.ndarray
        """

        self.vertices = np.array(vertices, dtype=float)
        """ (numpy.ndarray) Vertices. """

        self.faces = np.array(faces, dtype=int)
        """ (numpy.ndarray) Faces. """

        assert self.faces.shape[1] == 3

    @property
    def vertices(self):
        return vertices

    @property
    def faces(self):
        return faces

    def extents(self):
        """
        Get the extents.

        :return: (min_x, min_y, min_z), (max_x, max_y, max_z)
        :rtype: (float, float, float), (float, float, float)
        """

        min = [0] * 3
        max = [0] * 3

        for i in range(3):
            min[i] = np.min(self.vertices[:, i])
            max[i] = np.max(self.vertices[:, i])

        return tuple(min), tuple(max)

    def scale(self, scales):
        """
        Scale the mesh in all dimensions.

        :param scales: tuple of length 3 with scale for (x, y, z)
        :type scales: (float, float, float)
        """

        assert len(scales) == 3

        for i in range(3):
            self.vertices[:, i] *= scales[i]

    def translate(self, translation):
        """
        Translate the mesh.

        :param translation: translation as (x, y, z)
        :type translation: (float, float, float)
        """

        assert len(translation) == 3

        for i in range(3):
            self.vertices[:, i] += translation[i]


def transform_raw_mesh(input_file, output_file, padding, dims):
    input_format = input_file[input_file.rfind('.')+1:]
    output_format = output_file[output_file.rfind('.') + 1:]

    if(input_format == 'off'):
        vertices, faces = formatHandler.read_off(input_file)
    elif(input_format == 'ply'):
        vertices, faces = formatHandler.read_ply(input_file)

    mesh = Mesh(vertices, faces)

    # Get extents of model.
    min, max = mesh.extents()
    total_min = np.min(np.array(min))
    total_max = np.max(np.array(max))

    print('%s extents before %f - %f, %f - %f, %f - %f.' % (
        os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))

    # Set the center (although this should usually be the origin already).
    centers = (
        (min[0] + max[0]) / 2,
        (min[1] + max[1]) / 2,
        (min[2] + max[2]) / 2
    )
    # Scales all dimensions equally.
    sizes = (
        total_max - total_min,
        total_max - total_min,
        total_max - total_min
    )
    translation = (
        -centers[0],
        -centers[1],
        -centers[2]
    )
    scales = (
        1 / (sizes[0] + 2 * padding * sizes[0]),
        1 / (sizes[1] + 2 * padding * sizes[1]),
        1 / (sizes[2] + 2 * padding * sizes[2])
    )

    scale = max(dims)

    mesh.translate(translation)
    mesh.scale(scales)

    mesh.translate((0.5, 0.5, 0.5))
    mesh.scale((scale, scale, scale))

    min, max = mesh.extents()
    print('%s extents after %f - %f, %f - %f, %f - %f.' % (
        os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))

    print(f"scales used : {scales}")
    print(f"translation used : {translation}")

    if output_format == 'off':
        formatHandler.write_off(output_file, mesh.vertices, mesh.faces)
    elif output_format == 'ply':
        formatHandler.write_ply(output_file, mesh.vertices, mesh.faces)


def transform(inp_dir, op_dir, output_format, padding, dims):

    if not os.path.exists(inp_dir):
        print('Input directory does not exist.')
        exit(1)

    if not os.path.exists(op_dir):
        os.makedirs(op_dir)
        print('Created output directory.')
    else:
        print('Output directory exists; potentially overwriting contents.')

    for filename in os.listdir(inp_dir):
        input_file = os.path.join(inp_dir, filename)
        output_file = os.path.join(op_dir,filename[:filename.rfind('.')+1] + output_format)

        transform_raw_mesh(input_file, output_file, padding, dims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transforms the raw mesh to lie into [0, H] x [0, W] x [0, D] corresponding to choosen resolution.')
    parser.add_argument('input', type=str, help='Directory containing input files (OFF/PLY)')
    parser.add_argument('output', type=str, help='Output Folder')
    parser.add_argument('--output_format', type=str, default='ply', help='Output Format (OFF/PLY)')
    parser.add_argument('--padding', type=float, default=0, help='Padding on each side.')
    parser.add_argument('--height', type=int, default=32, help='Height to scale to.')
    parser.add_argument('--width', type=int, default=32, help='Width to scale to.')
    parser.add_argument('--depth', type=int, default=32, help='Depth to scale to.')

    args = parser.parse_args()
    dims = [args.width, args.height, args.depth]

    transform(args.input, args.output, args.output_format, args.padding, dims)