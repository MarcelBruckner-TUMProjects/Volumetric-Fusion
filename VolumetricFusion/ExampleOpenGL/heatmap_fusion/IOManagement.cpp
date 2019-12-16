#include "IOManagement.h"
#include "IOManagement.h"

#include <thread>
#include <chrono>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>
#include <iterator>
#include <iostream>

#include "lodepng.h"
#include "tinyply.h"

//#include <Windows.h>

using namespace tinyply;

namespace heatmap_fusion {
	namespace io_management {

		TriMesh loadMeshPly(const std::string& meshPath) {
			bool bPrintExceptions = false;

			try {
				std::ifstream ss(meshPath, std::ios::binary);
				if (ss.fail()) throw std::runtime_error("failed to open " + meshPath);

				PlyFile file;
				file.parse_header(ss);

				//for (auto c : file.get_comments()) std::cout << "Comment: " << c << std::endl;
				//for (auto e : file.get_elements()) {
				//	std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
				//	for (auto p : e.properties) std::cout << "\tproperty - " << p.name << " (" << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
				//}

				// Tinyply treats parsed data as untyped byte buffers. See below for examples.
				std::shared_ptr<PlyData> vertices, normals, faces, texcoords;

				// The header information can be used to programmatically extract properties on elements
				// known to exist in the header prior to reading the data. For brevity of this sample, properties 
				// like vertex position are hard-coded: 
				try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
				catch (const std::exception& e) { if (bPrintExceptions) std::cerr << "tinyply exception: " << e.what() << std::endl; }

				try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
				catch (const std::exception& e) { if (bPrintExceptions) std::cerr << "tinyply exception: " << e.what() << std::endl; }

				try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
				catch (const std::exception& e) { if (bPrintExceptions) std::cerr << "tinyply exception: " << e.what() << std::endl; }

				// Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
				// arbitrary ply files, it is best to leave this 0. 
				try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
				catch (const std::exception& e) { if (bPrintExceptions) std::cerr << "tinyply exception: " << e.what() << std::endl; }

				file.read(ss);
				
				TriMesh mesh;

				if (vertices && vertices->count > 0) {
					const size_t numBytes = vertices->buffer.size_bytes();
					std::vector<float3> verts(vertices->count);
					std::memcpy(verts.data(), vertices->buffer.get(), numBytes);

					mesh.positions.allocate(vertices->count, true, false);
					for (int i = 0; i < vertices->count; i++) {
						float3 p = verts[i];
						mesh.positions.getElement(i, Type2Type<MemoryTypeCPU>()) = make_float4(p.x, p.y, p.z, 1.f);
					}

					mesh.positions.setUpdated(true, false);
				}

				if (normals && normals->count > 0) {
					const size_t numBytes = normals->buffer.size_bytes();
					std::vector<float3> verts(normals->count);
					std::memcpy(verts.data(), normals->buffer.get(), numBytes);

					mesh.normals.allocate(normals->count, true, false);
					for (int i = 0; i < normals->count; i++) {
						float3 p = verts[i];
						mesh.normals.getElement(i, Type2Type<MemoryTypeCPU>()) = make_float4(p.x, p.y, p.z, 1.f);
					}

					mesh.normals.setUpdated(true, false);
				}

				if (texcoords) {
					std::cout << "Skipping texture coordinates at ply loading." << std::endl;
				}

				if (faces && faces->count > 0) {
					const size_t numBytes = faces->buffer.size_bytes();
					std::vector<uint3> indices(faces->count);
					std::memcpy(indices.data(), faces->buffer.get(), numBytes);

					mesh.faceIndices.allocate(faces->count * 3, true, false);
					for (int i = 0; i < faces->count; i++) {
						uint3 idx = indices[i];
						mesh.faceIndices.h(3 * i + 0) = idx.x;
						mesh.faceIndices.h(3 * i + 1) = idx.y;
						mesh.faceIndices.h(3 * i + 2) = idx.z;
					}

					mesh.faceIndices.setUpdated(true, false);
				}

				return mesh;
			}
			catch (const std::exception& e) {
				std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
			}
		}

		Array2<uchar4> loadColorImage(const std::string& colorImagePath) {
			std::vector<unsigned char> image; //the raw pixels
			unsigned width, height;

			//decode
			unsigned error = lodepng::decode(image, width, height, colorImagePath);

			//if there's an error, display it
			if (error) std::cout << "Color decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

			Array2<uchar4> colorImage(width, height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int idx = x + y * width;
					colorImage(x, y) = make_uchar4(
						image[4*idx + 0],
						image[4*idx + 1],
						image[4*idx + 2],
						image[4*idx + 3]
					);
				}
			}

			return colorImage;
		}

		Array2<float> loadDepthImage(const std::string& depthImagePath) {
			std::vector<unsigned char> png;
			std::vector<unsigned char> image; //the raw pixels
			unsigned width, height;

			//width = 512;
			//height = 424;
			
			lodepng::State state;
			state.info_raw.colortype = LodePNGColorType::LCT_GREY;
			state.info_raw.bitdepth = 16;

			//char buf[256];
			//GetCurrentDirectoryA(256, buf);
			//cout << buf << endl;
			//cout << depthImagePath << endl;

			unsigned error = lodepng::load_file(png, depthImagePath); //load the image file with given filename
			if (!error) error = lodepng::decode(image, width, height, state, png);

			//if there's an error, display it
			if (error) std::cout << "Depth decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

			Array2<float> depthImage(width, height);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int idx = x + y * width;

					unsigned char a = image[2 * idx + 0];
					unsigned char b = image[2 * idx + 1];

					unsigned short depth = (((unsigned short)a) << 8) | b;
					float depthFloat = float(depth) / 1000.f;
					depthImage(x, y) = depthFloat;
				}
			}

			return depthImage;
		}

	} // namespace io_management 
} // namespace heatmap_fusion
