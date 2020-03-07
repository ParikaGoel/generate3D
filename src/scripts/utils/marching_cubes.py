import os
import h5py
import argparse
import numpy as np
import data_formats as formatHandler

skimage = None
mcubes = None

def read_sdf(filename):
    with open(filename, 'rb') as f:
        dims = np.fromfile(f, dtype=np.int32, count=3)
        res = np.fromfile(f, dtype=np.float32, count=1)
        grid2world = np.fromfile(f, dtype=np.float32, count=16)

        n_size = dims[0] * dims[1] * dims[2]
        sdf = np.fromfile(f, dtype=np.float32, count=n_size)
        pdf = np.fromfile(f, dtype=np.float32, count=n_size)

    sdf = np.asarray(sdf).reshape(dims[0], dims[1], dims[2])

    return sdf

try:
    from skimage import measure

    def marching_cubes_skimage(tensor):
        """
        Perform marching cubes using mcubes.

        :param tensor: input volume
        :type tensor: numpy.ndarray
        :return: vertices, faces
        :rtype: numpy.ndarray, numpy.ndarray
        """

        vertices, faces, normals, values = measure.marching_cubes_lewiner(tensor.transpose(1, 0, 2), 0)
        return vertices, faces

    print('Using skimage\'s marching cubes implementation.')
except ImportError:
    print('Could not find skimage, import skimage.measure failed.')
    print('If you use skimage, make sure to call voxelize with -mode=corner.')

try:
    import mcubes

    def marching_cubes_mcubes(tensor):
        """
        Perform marching cubes using mcubes.

        :param tensor: input volume
        :type tensor: numpy.ndarray
        :return: vertices, faces
        :rtype: numpy.ndarray, numpy.ndarray
        """

        return mcubes.marching_cubes(-tensor.transpose(1, 0, 2), 0)

    print('Using PyMCubes\'s marching cubes implementation.')
except ImportError:
    print('Could not find PyMCubes, import mcubes failed.')
    print('You can use the version at https://github.com/davidstutz/PyMCubes.')
    print('If you use the voxel_centers branch, you can use -mode=center, otherwise use -mode=corner.')

if mcubes == None and measure == None:
    print('Could not find any marching cubes implementation; aborting.')
    exit(1);

if __name__ == '__main__':

    input_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/02747177/ffe5f0ef45769204cb2a965e75be701c__0__.df"
    output = "/home/parika/WorkingDir/complete3D/Assets"
    tensor = read_sdf(input_file)

    # parser = argparse.ArgumentParser(description='Peform marching cubes.')
    # parser.add_argument('input', type=str, help='The input HDF5 file.')
    # parser.add_argument('output', type=str, help='Output directory for OFF files.')
    #
    # args = parser.parse_args()
    # if not os.path.exists(args.input):
    #     print('Input file does not exist.')
    #     exit(1)
    #
    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)
    #     print('Created output directory.')
    # else:
    #     print('Output directory exists; potentially overwriting contents.')
    #
    # tensor = read_hdf5(args.input)
    if len(tensor.shape) < 4:
        tensor = np.expand_dims(tensor, axis=0)

    for n in range(tensor.shape[0]):
        print('Minimum and maximum value: %f and %f. ' % (np.min(tensor[n]), np.max(tensor[n])))
        vertices, faces = marching_cubes_skimage(tensor[n])
        off_file = '%s/%d.off' % (args.output, n)
        formatHandler.write_off(off_file, vertices, faces)
        print('Wrote %s.' % off_file)

    print('Use MeshLab to visualize the created OFF files.')