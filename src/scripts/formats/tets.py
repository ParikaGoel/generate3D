import data_formats as format

if __name__=='__main__':
    file = "/home/parika/WorkingDir/complete3D/Assets/test/shapenet-data/models/model_normalized.obj"
    ply = "/home/parika/WorkingDir/complete3D/Assets/test/generated_model.ply"
    vertices, faces = format.read_obj(file);
    format.write_ply(ply, vertices, faces);