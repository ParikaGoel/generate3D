//SDFGen - A simple grid-based signed distance field (level set) generator for triangle meshes.
//Written by Christopher Batty (christopherbatty@yahoo.com, www.cs.columbia.edu/~batty)
//...primarily using code from Robert Bridson's website (www.cs.ubc.ca/~rbridson)
//This code is public domain. Feel free to mess with it, let me know if you like it.

#include "makelevelset3.h"
#include "config.h"

#include "LoaderVOX.h"
#include "LoaderMesh.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>

int main(int argc, char* argv[]) {

	if(argc != 6) {
		std::cout << "SDFGen - A utility for converting closed oriented triangle meshes into grid-based signed distance fields.\n";
		std::cout << "\nThe output file format is:";
		std::cout << "<ni> <nj> <nk>\n";
		std::cout << "<origin_x> <origin_y> <origin_z>\n";
		std::cout << "<dx>\n";
		std::cout << "<value_1> <value_2> <value_3> [...]\n\n";

		std::cout << "(ni,nj,nk) are the integer dimensions of the resulting distance field.\n";
		std::cout << "(origin_x,origin_y,origin_z) is the 3D position of the grid origin.\n";
		std::cout << "<dx> is the grid spacing.\n\n";
		std::cout << "<value_n> are the signed distance data values, in ascending order of i, then j, then k.\n";

		std::cout << "The output filename will match that of the input, with the OBJ suffix replaced with SDF.\n\n";

		std::cout << "Usage: SDFGen <filename> <dx> <padding> <is_unit_cube> <filename out>\n\n";
		std::cout << "Where:\n";
		std::cout << "\t<filename> specifies a Wavefront OBJ (text) file representing a *triangle* mesh (no quad or poly meshes allowed). File must use the suffix \".obj\".\n";
		std::cout << "\t<dx> specifies the length of grid cell in the resulting distance field.\n";
		std::cout << "\t<padding> specifies the number of cells worth of padding between the object bound box and the boundary of the distance field grid. Minimum is 1.\n\n";
		std::cout << "\t<Is unit cube?. Assumes unit cube [0,1] false/true.\n\n";
		std::cout << "\t<filename out. E.g. out.vox\n\n";

		exit(-1);
	}

	std::string filename(argv[1]);

	std::stringstream arg2(argv[2]);
	int dim;
	arg2 >> dim;

	std::stringstream arg3(argv[3]);
	int padding;
	arg3 >> padding;
	
	float dx = 1.0/(dim - 2*padding);
	
	std::stringstream arg4(argv[4]);
	int is_unit_cube;
	arg4 >> is_unit_cube;
	
	std::string filename_out(argv[5]);

	//start with a massive inside out bound box.
	Vec3f min_box(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()), 
		  max_box(-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max());

	Mesh mesh;
	load_mesh(filename, mesh);
	std::vector<Vec3f> vertList;
	std::vector<Vec3ui> faceList;
	for (int i = 0; i < mesh.V.cols(); i++) {
		auto v = mesh.V.col(i);
		Vec3f point(v(0), v(1), v(2));
		update_minmax(point, min_box, max_box);
		vertList.push_back(point);
	}
	for (int i = 0; i < mesh.F.cols(); i++) {
		auto f = mesh.F.col(i);
		faceList.push_back(Vec3ui(f(0), f(1), f(2)));
	}

	if (is_unit_cube) {
		min_box = Vec3f(-1,-1,-1)*0.5;
		max_box = Vec3f(1,1,1)*0.5;
	}


	std::cout << "Read in " << vertList.size() << " vertices and " << faceList.size() << " faces." << std::endl;

	//Add padding around the box.
	Vec3f unit(1,1,1);
	min_box -= padding*dx*unit;
	max_box += padding*dx*unit;
	Vec3ui sizes = Vec3ui((max_box - min_box)/dx);

	std::cout << "padding: " << padding << " dx: " << dx << std::endl;

	std::cout << "Bound box size: (" << min_box << ") to (" << max_box << ") with dimensions " << sizes << "." << std::endl;

	Array3f phi_grid;
	make_level_set3(faceList, vertList, min_box, dx, sizes[0], sizes[1], sizes[2], phi_grid);
	assert(phi_grid.ni == dim && phi_grid.nj == dim && phi_grid.nk == dim);

	Vox vox;
	vox.dims = Eigen::Vector3i(phi_grid.ni, phi_grid.nj, phi_grid.nk);
	vox.res = 1.0;
	float dfix = 1.0/dim;
	vox.grid2world = Eigen::Matrix4f::Identity();
	vox.grid2world.block(0,0,3,3) *= dx;
	if(padding != 0)
	    vox.grid2world.block(0,3,3,1) = -0.5*Eigen::Vector3f(1,1,1) - Eigen::Vector3f::Constant(dfix); // <-- padding introduces offset
	else
	    vox.grid2world.block(0,3,3,1) = -0.5*Eigen::Vector3f(1,1,1);

	vox.grid2world = vox.grid2world.transpose().eval(); // <-- make row-major
	std::cout << "Grid to World : " << vox.grid2world << "\n";
	vox.sdf.resize(phi_grid.a.size());
	for(unsigned int i = 0; i < phi_grid.a.size(); ++i) {
		vox.sdf[i] = phi_grid.a[i]*dim;
	}
	save_vox(filename_out, vox);


	return 0;
}
