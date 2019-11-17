//
// Created by parika on 14.11.19.
//

#ifndef COMPLETE3D_MESH_LOADER_H
#define COMPLETE3D_MESH_LOADER_H

#include <string>
#include <iostream>

#include <global_structs.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

inline static bool load_obj_with_tinyobjloader(std::string filename, mesh &mesh) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

        if (!err.empty()) {
            std::cout<<err;
        }

        if (!ret) {
            std::cout<<filename;
            return false;
        }

        if (shapes.size() == 0) {
            std::cout<<"err: # of shapes are zero.\n";
            return false;
        }

        // Loop over vertices
        for(size_t v=0; v < attrib.vertices.size()/3;v++ ){
            vertex new_v;
            new_v.vx = attrib.vertices[3*v+0]; // x coord
            new_v.vy = attrib.vertices[3*v+1]; // y coord
            new_v.vz = attrib.vertices[3*v+2]; // z coord
            mesh.vertices.emplace_back(new_v);
        }

        // Loop over shapes
        for(size_t s = 0;s< shapes.size(); s++){
            // Loop over faces
            size_t vertex_index_offset = 0;
            size_t num_faces = shapes[s].mesh.num_face_vertices.size();
            for(size_t f =0 ; f < num_faces; f++){
                struct face new_face;
                new_face.num_vertices = shapes[s].mesh.num_face_vertices[f];

                // Loop over vertices
                for(size_t v =0 ; v < new_face.num_vertices; v++){
                    new_face.vertex_indices.emplace_back(shapes[s].mesh.indices[vertex_index_offset+v].vertex_index);
                    vertex_index_offset++;
                }
                mesh.faces.emplace_back(new_face);
            }
        }

        std::cout<<"file: " << filename << "\n" ;
        std::cout<<"n-faces: " << mesh.faces.size() << "\n";
        std::cout<<"n-vertices: " << mesh.vertices.size() << "\n";
}

inline static bool load_obj(std::string filename, mesh &mesh){
    std::cout << "Reading data.\n";

    std::ifstream infile(filename);
    if(!infile) {
        std::cerr << "Failed to open. Terminating.\n";
        exit(-1);
    }

    int ignored_lines = 0;
    std::string line;
    while(!infile.eof()) {
        std::getline(infile, line);

        //.obj files sometimes contain vertex normals indicated by "vn"
        if(line.substr(0,1) == std::string("v") && line.substr(0,2) != std::string("vn") && line.substr(0,2) != std::string("vt") && line.substr(0,2) != std::string("vp")){
            std::stringstream data(line);
            char c;
            vertex point;
            data >> c >> point.vx >> point.vy >> point.vz;
            mesh.vertices.emplace_back(point);
        }
        else if(line.substr(0,1) == std::string("f")) {
            std::stringstream data(line);
            struct face triangle;
            char c;
            int v0,v1,v2;
            triangle.num_vertices = 3;
            data >> c >> v0 >> v1 >> v2;
            triangle.vertex_indices.push_back(v0-1);
            triangle.vertex_indices.push_back(v1-1);
            triangle.vertex_indices.push_back(v2-1);
            mesh.faces.emplace_back(triangle);
        }
        else {
            ++ignored_lines;
        }
    }
    infile.close();

    if(ignored_lines > 0)
        std::cout << "Warning: " << ignored_lines << " lines were ignored since they did not contain faces or vertices.\n";

    std::cout << "Read in " << mesh.vertices.size() << " vertices and " << mesh.faces.size() << " faces." << std::endl;
}

inline static bool load_mesh(std::string filename, mesh &mesh){
    if (filename.find(".obj") != std::string::npos)
        load_obj_with_tinyobjloader(filename,mesh);
    else{
        fprintf(stderr, "Error: mesh format not known.\n");
        exit(1);
    }
    assert(mesh.vertices.size() > 0 && mesh.faces.size() > 0 && "Error loading mesh.");
}

#endif //COMPLETE3D_MESH_LOADER_H
