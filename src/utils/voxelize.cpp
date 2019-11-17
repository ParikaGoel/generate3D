#include <iostream>
#include <limits>
#include <fstream>

#include <Eigen/Dense>
#include <aabb_triangle_overlap.h>
#include <voxelize.h>

/** \brief Compute triangle box intersection.
 * \param[in] min defining voxel
 * \param[in] max defining voxel
 * \param[in] v1 first vertex
 * \param[in] v2 second vertex
 * \param[in] v3 third vertex
 * \return intersects
 */
bool triangle_box_intersection(const Eigen::Vector3f &min, Eigen::Vector3f &max, const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, const Eigen::Vector3f &v3) {
    float half_size[3] = {
            (max(0) - min(0))/2.0f,
            (max(1) - min(1))/2.0f,
            (max(2) - min(2))/2.0f
    };

    float center[3] = {
            max(0) - half_size[0],
            max(1) - half_size[1],
            max(2) - half_size[2]
    };

    float vertices[3][3] = {{v1(0), v1(1), v1(2)}, {v2(0), v2(1), v2(2)}, {v3(0), v3(1), v3(2)}};
    return triBoxOverlap(center, half_size, vertices);
}

/** \brief Voxelize the given mesh into an occupancy grid.
   * \param[out] occ_grid volume to fill
   */
void voxelize_to_occ_grid(mesh& mesh, voxel& occ_grid) {
    size_t width = occ_grid.dims[0];
    size_t height = occ_grid.dims[1];
    size_t depth = occ_grid.dims[2];
    float dx = occ_grid.cell_dist;
    size_t n_cells = width * height * depth;
    occ_grid.occ_grid_vals.resize(n_cells,false);
    size_t counter = 0;
    size_t n_voxels = 0;

    for(float w = occ_grid.bottom_left[0]; w <= occ_grid.top_right[0] && counter < n_cells; w += dx){
        for(float h = occ_grid.bottom_left[1]; h <= occ_grid.top_right[1] && counter < n_cells; h += dx){
            for(float d = occ_grid.bottom_left[2]; d <= occ_grid.top_right[2] && counter < n_cells; d += dx){
                Eigen::Vector3f min(w, h, d);
                Eigen::Vector3f max(w + dx, h + dx, d + dx);

                for(const auto& face: mesh.faces){
                    Eigen::Vector3f v1(
                            mesh.vertices[face.vertex_indices[0]].vx,
                            mesh.vertices[face.vertex_indices[0]].vy,
                            mesh.vertices[face.vertex_indices[0]].vz);

                    Eigen::Vector3f v2(
                            mesh.vertices[face.vertex_indices[1]].vx,
                            mesh.vertices[face.vertex_indices[1]].vy,
                            mesh.vertices[face.vertex_indices[1]].vz);

                    Eigen::Vector3f v3(
                            mesh.vertices[face.vertex_indices[2]].vx,
                            mesh.vertices[face.vertex_indices[2]].vy,
                            mesh.vertices[face.vertex_indices[2]].vz);

                    bool overlap = triangle_box_intersection(min, max, v1, v2, v3);
                    if (overlap) {
                        occ_grid.occ_grid_vals[counter]=true;
                        n_voxels++;
                        break;
                    }
                }
                counter++;
            }
        }
    }
    std::cout<<"Final counter value: " << counter << "\n";
    std::cout<<"Occupied voxels: " << n_voxels << "\n";
}

int voxelize_mesh(voxel& vox, mesh& mesh,
        size_t dim, size_t padding,
        bool is_unit_cube, std::string out_file){

	if(padding < 1) padding = 1;

    float dx = 1.0/(dim - 2*padding);

	//start with a massive inside out bound box.
	Eigen::Vector3f min_box(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()),
		  max_box(-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max());

	// update the bounding box depending on vertices
	for(const auto& vertex : mesh.vertices){
	    Eigen::Vector3f point(vertex.vx, vertex.vy, vertex.vz);
	    update_minmax(point, min_box, max_box);
	}

    if (is_unit_cube) {
        min_box = Eigen::Vector3f(-1,-1,-1)*0.5;
        max_box = Eigen::Vector3f(1,1,1)*0.5;
    }

	//Add padding around the box.
	Eigen::Vector3f unit(1,1,1);
	min_box -= padding * dx * unit;
    max_box += padding * dx * unit;

    Eigen::Vector3f range((max_box - min_box)/dx);

	Eigen::Vector3i sizes((int)range[0],(int)range[1],(int)range[2]);

	std::cout << "padding: " << padding << " dx: " << dx << std::endl;

	std::cout << "Bound box size: (" << min_box << ") to (" << max_box << ") with dimensions " << sizes << "." << std::endl;

	vox.dims = {dim, dim, dim};
	vox.bottom_left = min_box;
	vox.top_right = max_box;
	vox.cell_dist = dx;

	voxelize_to_occ_grid(mesh,vox);
	save_vox("./out.vox",vox);

	std::cout<<"Voxelization complete\n";

	return 0;
}

void save_vox(std::string filename, voxel &vox) {
    std::ofstream f;
    f.open(filename, std::ofstream::out | std::ios::binary);
    assert(f.is_open());
    f.write((char*)vox.dims.data(), 3*sizeof(size_t));
    f.write((char*)&vox.cell_dist, sizeof(float));
    size_t n_size = vox.dims[0]*vox.dims[1]*vox.dims[2];
    for(const auto& val : vox.occ_grid_vals)
        f << val << "\n";
    f.close();
}
