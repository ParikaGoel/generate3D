import struct
import os
import torch
import plyfile
import numpy as np
# import SGNN.util.marching_cubes.marching_cubes as mc
# import SGNN.util.constants as constants
# from tqdm import tqdm

def load_sdf_file(target_path):
    # fin = open(input_path, 'rb')
    # dimx = struct.unpack('Q', fin.read(8))[0]
    # dimy = struct.unpack('Q', fin.read(8))[0]
    # dimz = struct.unpack('Q', fin.read(8))[0]
    # voxelsize = struct.unpack('f', fin.read(4))[0]
    # world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    # world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    # # input data
    # num = struct.unpack('Q', fin.read(8))[0]
    # input_locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    # input_locs = np.asarray(input_locs, dtype=np.int32).reshape([num, 3])
    # input_locs = np.flip(input_locs,1).copy() # convert to zyx ordering
    # input_sdfs = struct.unpack('f'*num, fin.read(num*4))
    # input_sdfs = np.asarray(input_sdfs, dtype=np.float32)
    # input_sdfs /= voxelsize
    
    # target data
    fin = open(target_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    # assert(struct.unpack('Q', fin.read(8))[0] == dimx)
    # assert(dimy == struct.unpack('Q', fin.read(8))[0])
    # assert(dimz == struct.unpack('Q', fin.read(8))[0])
    # assert(voxelsize == struct.unpack('f', fin.read(4))[0])
    # struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    num = struct.unpack('Q', fin.read(8))[0]
    target_locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    target_locs = np.asarray(target_locs, dtype=np.int32).reshape([num, 3])
    target_locs = np.flip(target_locs,1).copy() # convert to zyx ordering
    target_sdfs = struct.unpack('f'*num, fin.read(num*4))
    target_sdfs = np.asarray(target_sdfs, dtype=np.float32)
    target_sdfs /= voxelsize
    target_sdfs = sparse_to_dense_np(target_locs, target_sdfs[:,np.newaxis], dimx, dimy, dimz, -float('inf'))

    return target_sdfs, [dimz, dimy, dimx], world2grid
    # return [input_locs, input_sdfs], target_sdfs, [dimz, dimy, dimx], world2grid

def load_sdf_colors_file(input_path, target_path, rescale_factor, use_completed_geometry):
    trunc = 3.
    fin = open(input_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    # input data
    num = struct.unpack('Q', fin.read(8))[0]
    input_locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    input_locs = np.asarray(input_locs, dtype=np.int32).reshape([num, 3])
    input_locs = np.flip(input_locs,1).copy() # convert to zyx ordering
    input_sdfs = struct.unpack('f'*num, fin.read(num*4))
    input_sdfs = np.asarray(input_sdfs, dtype=np.float32)
    input_sdfs /= voxelsize
    input_colors_coded = struct.unpack('i'*num, fin.read(num*4))
    input_colors_coded = np.asarray(input_colors_coded, dtype=np.int32)
    input_colors = np.zeros([input_colors_coded.shape[0], 3], dtype=np.float32)
    input_colors[:, 0] = (((input_colors_coded.flatten() // 256) // 256) / 255) * 2 - 1
    input_colors[:, 1] = (((input_colors_coded.flatten() // 256) % 256) / 255) * 2 - 1
    input_colors[: ,2] = ((input_colors_coded.flatten() % 256) / 255) * 2 - 1

    # target data
    fin = open(target_path, 'rb')
    assert(struct.unpack('Q', fin.read(8))[0] == dimx)
    assert(dimy == struct.unpack('Q', fin.read(8))[0])
    assert(dimz == struct.unpack('Q', fin.read(8))[0])
    assert(voxelsize == struct.unpack('f', fin.read(4))[0])
    struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    num = struct.unpack('Q', fin.read(8))[0]
    target_locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    target_locs = np.asarray(target_locs, dtype=np.int32).reshape([num, 3])
    target_locs = np.flip(target_locs,1).copy() # convert to zyx ordering
    target_sdfs = struct.unpack('f'*num, fin.read(num*4))
    target_sdfs = np.asarray(target_sdfs, dtype=np.float32)
    target_sdfs /= voxelsize
    target_sdfs = sparse_to_dense_np(target_locs, target_sdfs[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
    target_colors_coded = struct.unpack('i'*num, fin.read(num*4))
    target_colors_coded = np.asarray(target_colors_coded, dtype=np.int32)
    target_colors = sparse_to_dense_colors_flat(target_locs, target_colors_coded[:,np.newaxis], dimx, dimy, dimz, -1)

    if use_completed_geometry:
        input_colors_dense = sparse_to_dense_colors(input_locs, input_colors, dimz, dimy, dimx, -1)
        input_locs = np.stack(np.where(np.abs(target_sdfs) < trunc), 1)
        input_sdfs = target_sdfs[input_locs[:, 0], input_locs[:, 1], input_locs[:, 2]]
        input_colors = input_colors_dense[input_locs[:, 0], input_locs[:, 1], input_locs[:, 2], :]

    if rescale_factor != 1:
        input_sdf_dense = sparse_to_dense_np(input_locs, input_sdfs[:, np.newaxis], dimz, dimy, dimx, -float('inf'))
        input_sdf_dense = torch.nn.functional.interpolate(torch.from_numpy(input_sdf_dense).unsqueeze(0).unsqueeze(0), scale_factor=rescale_factor) * rescale_factor
        input_sdf_dense = input_sdf_dense[0, 0].numpy()
        input_colors_dense = sparse_to_dense_colors(input_locs, input_colors, dimz, dimy, dimx, -1)
        input_colors_dense = torch.nn.functional.interpolate(torch.from_numpy(input_colors_dense).permute(3, 0, 1, 2).contiguous().unsqueeze(0).float(), scale_factor=rescale_factor)
        input_colors_dense = input_colors_dense[0].permute(1, 2, 3, 0).contiguous().numpy()
        target_sdfs = torch.nn.functional.interpolate(torch.from_numpy(target_sdfs).unsqueeze(0).unsqueeze(0), scale_factor=rescale_factor) * rescale_factor
        target_sdfs = target_sdfs[0, 0].numpy()
        target_colors = torch.nn.functional.interpolate(torch.from_numpy(target_colors).permute(3, 0, 1, 2).contiguous().unsqueeze(0).float(), scale_factor=rescale_factor)
        target_colors = target_colors[0].permute(1, 2, 3, 0).contiguous().numpy()
        input_locs = np.stack(np.where(np.abs(input_sdf_dense) < trunc), 1)
        input_sdfs = input_sdf_dense[input_locs[:, 0], input_locs[:, 1], input_locs[:, 2]]
        input_colors = input_colors_dense[input_locs[:, 0], input_locs[:, 1], input_locs[:, 2], :]

    return [input_locs, input_sdfs, input_colors], [target_sdfs, target_colors], [dimz, dimy, dimx], world2grid


def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2],:] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


def sparse_to_dense_colors_flat(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 3
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=np.float32)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2], 0] = (((values.flatten() // 256) // 256) / 255) * 2 - 1
    dense[locs[:,0], locs[:,1], locs[:,2], 1] = (((values.flatten() // 256) % 256) / 255) * 2 - 1
    dense[locs[:,0], locs[:,1], locs[:,2], 2] = ((values.flatten() % 256) / 255) * 2 - 1
    return dense.reshape([dimz, dimy, dimx, nf_values])


def sparse_to_dense_colors(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 3
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=np.float32)
    dense.fill(default_val)
    dense[locs[:,0], locs[:,1], locs[:,2], :] = values
    return dense.reshape([dimz, dimy, dimx, nf_values])


def visualize_occ_as_points(sdf, thresh, output_file, transform=None, thresh_max = float('inf')):
    # collect verts from sdf
    verts = []
    for z in range(sdf.shape[0]):
        for y in range(sdf.shape[1]):
            for x in range(sdf.shape[2]):
                val = abs(sdf[z, y, x])
                if val > thresh and val < thresh_max:
                    verts.append(np.array([x, y, z]) + 0.5)  # center of voxel
    if len(verts) == 0:
        #print('warning: no valid occ points for %s' % output_file)
        return
    verts = np.stack(verts)
    visualize_points(verts, output_file, transform)

def visualize_sparse_locs_as_points(locs, output_file, transform=None, color=None):
    # collect verts from sdf
    verts = locs[:,:3]
    colors = None if color is None else color[:,:3]
    if len(verts) == 0:
        print('warning: no valid occ points for %s' % output_file)
        return
    verts = np.stack(verts).astype(np.float32)
    verts = verts[:,::-1] + 0.5
    if color is not None:
        colors = np.stack(colors).astype(np.float32) / 255.0
    visualize_points(verts, output_file, transform, colors)

def visualize_sparse_sdf_as_points(sdf_locs, sdf_vals, iso, output_file, transform=None, sdf_color=None):
    # collect verts from sdf
    mask = np.abs(sdf_vals) < iso
    verts = sdf_locs[:,:3][mask]
    colors = None
    if sdf_color is not None:
        colors = sdf_color[:,:3][mask]
    if len(verts) == 0:
        print('warning: no valid sdf points for %s' % output_file)
        return
    verts = np.stack(verts).astype(np.float32)
    verts = verts[:,::-1] + 0.5
    if colors is not None:
        colors = np.stack(colors).astype(np.float32)
    visualize_points(verts, output_file, transform, colors)

def visualize_points(points, output_file, transform=None, colors=None):
    verts = points if points.shape[1] == 3 else np.transpose(points)
    if transform is not None:
        x = np.ones((verts.shape[0], 4))
        x[:, :3] = verts
        x = np.matmul(transform, np.transpose(x))
        x = np.transpose(x)
        verts = np.divide(x[:, :3], x[:, 3, None])

    ext = os.path.splitext(output_file)[1]
    if colors is not None:
        colors = np.clip(colors, 0, 1)
    if colors is not None or ext == '.obj':
        output_file = os.path.splitext(output_file)[0] + '.obj'
        num_verts = len(verts)
        with open(output_file, 'w') as f:
            for i in range(num_verts):
                v = verts[i]
                if colors is None:
                    f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                else:
                    f.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], colors[i,0], colors[i,1], colors[i,2]))
    elif ext == '.ply':
        verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = plyfile.PlyElement.describe(verts,'vertex')
        plyfile.PlyData([el]).write(output_file)
    else:
        raise

def make_scale_transform(scale):
    if isinstance(scale, int) or isinstance(scale, float):
        scale = [scale, scale, scale]
    assert( len(scale) == 3 )
    transform = np.eye(4, 4)
    for k in range(3):
        transform[k,k] = scale[k]
    return transform


def export_sdf_as_mesh(output_path, names, inputs, target_for_sdf, target_for_color, target_for_occs, output_sdf, output_color, output_occs, world2grids, truncation, thresh=3):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    export_mesh_colors = True
    if target_for_color is None:
        export_mesh_colors = False
    if output_occs is not None:
        num_hierarchy_levels = len(output_occs)
        factors = [1] * num_hierarchy_levels
        for h in range(num_hierarchy_levels-2, -1, -1):
            factors[h] = factors[h+1] * 2
    dims = np.max(output_sdf[0][0],0)+1 if target_for_sdf is None else target_for_sdf.shape[2:]
    isovalue = 0
    trunc = truncation - 0.1
    ext = '.ply'

    for k in range(len(names)):
        name = names[k]
        subdir = os.path.join(output_path, name)
        os.makedirs(subdir, exist_ok=True)
        mask = inputs[0][:,-1] == k
        locs = inputs[0][mask]
        feats = inputs[1][mask]
        input = sparse_to_dense_np(locs[:, :-1], feats, dims[2], dims[1], dims[0], -float('inf'))
        input_colors = None
        if export_mesh_colors:
            colors = inputs[2][mask]
            input_colors = torch.from_numpy(((sparse_to_dense_colors(locs, colors, dims[2], dims[1], dims[0], 0)/2 + 0.5) * 255).astype(np.uint8))
            visualize_sparse_sdf_as_points(locs, feats.flatten(), thresh, os.path.join(subdir, 'input_occ' + ext), sdf_color=np.clip(colors / 2 + 0.5, 0, 1))
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        mc.marching_cubes(torch.from_numpy(input), input_colors, isovalue=isovalue, truncation=trunc, thresh=10, output_filename=os.path.join(subdir, 'input_mesh' + ext))
        if target_for_occs is not None and output_occs is not None:
            for h in range(num_hierarchy_levels):
                transform = make_scale_transform(factors[h])
                visualize_occ_as_points(target_for_occs[h][k,0] == 1, 0.5, os.path.join(subdir, f'target_{h:02d}{ext}'), transform, thresh_max=1.5)
                visualize_sparse_locs_as_points(output_occs[h][k], os.path.join(subdir, f'pred_{h:02d}{ext}'), transform)
        if output_sdf[k] is not None:
            locs = output_sdf[k][0][:,:3]
            pred_color_dense = None
            if export_mesh_colors:
                pred_color_dense = torch.from_numpy(((sparse_to_dense_colors(locs, output_color[k][1], dims[2], dims[1], dims[0], 0) / 2 + 0.5) * 255).astype(np.uint8))
                mask = np.abs(output_sdf[k][1]) <= 1
                visualize_sparse_locs_as_points(locs[mask], os.path.join(subdir, 'pred_occ.obj'), None, (np.clip(output_color[k][1][mask] * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8))
            pred_sdf_dense = sparse_to_dense_np(locs, output_sdf[k][1][:,np.newaxis], dims[2], dims[1], dims[0], -float('inf'))
            mc.marching_cubes(torch.from_numpy(pred_sdf_dense), pred_color_dense, isovalue=isovalue, truncation=trunc, thresh=10, output_filename=os.path.join(subdir, 'pred_mesh' + ext))
        if target_for_sdf is not None:
            target_sdf = target_for_sdf[k, 0]
            target_color = None
            if export_mesh_colors:
                target_color = (np.clip((np.transpose(target_for_color[k, :, :, :], [1, 2, 3, 0]) / 2 + 0.5), 0, 1) * 255).astype(np.uint8)
                visualize_sparse_locs_as_points(locs, os.path.join(subdir, 'target_occ.obj'), None, target_color[locs[:, 0], locs[:, 1], locs[:, 2]])
            mc.marching_cubes(torch.from_numpy(target_sdf), torch.from_numpy(target_color), isovalue=isovalue, truncation=trunc, thresh=10, output_filename=os.path.join(subdir, 'target_mesh' + ext))


def export_sdf_to_mesh_min(input_sdf_path, output_mesh_path):
    fin = open(input_sdf_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    # input data
    num = struct.unpack('Q', fin.read(8))[0]
    input_locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
    input_locs = np.asarray(input_locs, dtype=np.int32).reshape([num, 3])
    input_locs = np.flip(input_locs, 1).copy()  # convert to zyx ordering
    input_sdfs = struct.unpack('f' * num, fin.read(num * 4))
    input_sdfs = np.asarray(input_sdfs, dtype=np.float32)
    input_sdfs /= voxelsize
    input_sdfs = sparse_to_dense_np(input_locs, input_sdfs[:, np.newaxis], dimx, dimy, dimz, -float('inf'))
    input_colors_coded = struct.unpack('i' * num, fin.read(num * 4))
    input_colors_coded = np.asarray(input_colors_coded, dtype=np.int32)
    input_colors = (sparse_to_dense_colors_flat(input_locs, input_colors_coded[:, np.newaxis], dimx, dimy, dimz,0) * 256).astype(np.uint8)
    mc.marching_cubes(torch.from_numpy(input_sdfs), torch.from_numpy(input_colors), isovalue=0, truncation=3, thresh=10, output_filename=output_mesh_path)


def export_sdf_to_mesh_min_angelas_data(input_sdf_path, output_mesh_path):
    fin = open(input_sdf_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    # input data
    num = struct.unpack('Q', fin.read(8))[0]
    input_locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
    input_locs = np.asarray(input_locs, dtype=np.int32).reshape([num, 3])
    input_locs = np.flip(input_locs, 1).copy()  # convert to zyx ordering
    input_sdfs = struct.unpack('f' * num, fin.read(num * 4))
    input_sdfs = np.asarray(input_sdfs, dtype=np.float32)
    input_sdfs /= voxelsize
    input_sdfs = sparse_to_dense_np(input_locs, input_sdfs[:, np.newaxis], dimx, dimy, dimz, -float('inf'))
    num_known = struct.unpack('Q', fin.read(8))[0]
    fin.read(num_known)
    num_color = struct.unpack('Q', fin.read(8))[0]
    assert num_color == dimx * dimy * dimz
    input_colors = struct.unpack('B' * num_color * 3, fin.read(num_color * 3))
    input_colors = np.asarray(input_colors, dtype=np.uint8).reshape([dimz, dimy, dimx, 3])
    mc.marching_cubes(torch.from_numpy(input_sdfs), torch.from_numpy(input_colors), isovalue=0, truncation=3, thresh=10, output_filename=output_mesh_path)


def export_all_sdfs_to_meshes(filepath, outputpath):
    os.makedirs(outputpath, exist_ok=True)
    # root_sdf = '/mnt/sorona_angela_raid/data/matterport/completion_blocks_2cm_hierarchy/individual_96-96-160'
    root_sdf = '/media/nihalsid/OSDisk/Users/ga83fiz/nihalsid/ShapeNetSamples/sdfs'
    with open(filepath, "r") as fptr:
        sdfs = [x.strip() for x in fptr.readlines() if x.strip() != '']
    for sdf in tqdm(sdfs):
        sdf_path = os.path.join(root_sdf, sdf+".sdf")
        output_path = os.path.join(outputpath, sdf+".obj")
        if os.path.exists(sdf_path):
            export_sdf_to_mesh_min(sdf_path, output_path)


def export_voxel_occupancy(input_path, output_path):
    import h5py
    fin = open(input_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    # input data
    num = struct.unpack('Q', fin.read(8))[0]
    input_locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
    input_locs = np.asarray(input_locs, dtype=np.int32).reshape([num, 3])
    input_locs = np.flip(input_locs, 1).copy()  # convert to zyx ordering
    input_sdfs = struct.unpack('f' * num, fin.read(num * 4))
    input_sdfs = np.asarray(input_sdfs, dtype=np.float32)
    input_sdfs /= voxelsize
    input_sdfs = sparse_to_dense_np(input_locs, input_sdfs[:, np.newaxis], dimx, dimy, dimz, -float('inf'))
    num_known = struct.unpack('Q', fin.read(8))[0]
    fin.read(num_known)
    num_color = struct.unpack('Q', fin.read(8))[0]
    assert num_color == dimx * dimy * dimz
    input_colors = struct.unpack('B' * num_color * 3, fin.read(num_color * 3))
    input_colors = np.asarray(input_colors, dtype=np.float32).reshape([dimz, dimy, dimx, 3]) / 255.0

    rescale_factor = 0.5
    trunc = 1
    input_sdfs = torch.nn.functional.interpolate(torch.from_numpy(input_sdfs).unsqueeze(0).unsqueeze(0), scale_factor=rescale_factor) * rescale_factor
    input_sdfs = input_sdfs[0, 0].numpy()
    input_colors = torch.nn.functional.interpolate(torch.from_numpy(input_colors).permute(3, 0, 1, 2).contiguous().unsqueeze(0).float(), scale_factor=rescale_factor)
    input_colors = input_colors[0].permute(1, 2, 3, 0).contiguous().numpy()
    input_locs = np.stack(np.where(np.abs(input_sdfs) >= trunc), 1)
    input_colors[input_locs[:, 0], input_locs[:, 1], input_locs[:, 2], :] = -1
    fin.close()
    #input_colors = input_colors.transpose((2, 1, 0, 3))
    input_colors = input_colors.transpose((2, 1, 0, 3))
    #mc.marching_cubes(torch.from_numpy(input_sdfs), torch.from_numpy((input_colors*255).astype(np.uint8)), isovalue=0, truncation=3, thresh=10, output_filename='export_mesh.obj')
    ctr = 0

    h5_fout = h5py.File(output_path, "w")
    h5_fout.create_dataset(
        'data', data=input_colors,
        compression='gzip', compression_opts=4,
        dtype='float32')
    h5_fout.flush()
    h5_fout.close()

    '''
    with open("export.obj", "w") as fptr:
        for di in range(input_colors.shape[0]):
            for dj in range(input_colors.shape[1]):
                for dk in range(input_colors.shape[2]):
                    if input_colors[di, dj, dk, 0] != -1:
                        fptr.write('v %f %f %f %f %f %f\n' % (dk, dj, di, int(input_colors[di, dj, dk, 0] * 255.0), int(input_colors[di, dj, dk, 1] * 255.0), int(input_colors[di, dj, dk, 2] * 255.0)))
                        ctr += 1
    '''
    #print('vox: ', ctr)


# if __name__ == '__main__':
#     import sys
#     # export_all_sdfs_to_meshes('/home/yawar/matterport-chunks/'+sys.argv[1], '/mnt/sorona_angela_raid/matterport_chunks_mesh')
#     # export_all_sdfs_to_meshes('/media/nihalsid/OSDisk/Users/ga83fiz/nihalsid/ShapeNetSamples/sdfs/list.txt', '/media/nihalsid/OSDisk/Users/ga83fiz/nihalsid/ShapeNetSamples/sdfs/')
#     in_path = "/mnt/sorona_angela_raid/data/matterport/completion_blocks_2cm_hierarchy/individual_96-96-160/"
#     out_path = "/mnt/sorona_angela_raid/PiFU_matterport/color_volumes/"
#     all_sdfs = [x.split(".")[0] for x in os.listdir(in_path)]
#     part_len = len(all_sdfs) // 20 + 1
#     p = int(sys.argv[1])
#     for sdf in tqdm(all_sdfs[p * part_len:(p+1) * part_len]):
#         export_voxel_occupancy(os.path.join(in_path, sdf+".sdf"), os.path.join(out_path, sdf+".h5"))

if __name__ == '__main__':
    sdf_file = "/home/parika/WorkingDir/complete3D/Assets/yawar/1a6f615e8b1b5ae4dbbc9440457e303e__0__.sdf"
    target_sdfs, [dimz, dimy, dimx], world2grid = load_sdf_file(sdf_file)
    print(target_sdfs.shape)