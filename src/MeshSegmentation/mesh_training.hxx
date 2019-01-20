#pragma once
namespace MeshSegmentation
{
void train_mesh_segmentation(const char* _folder);
void apply_mesh_segmentation(const char* _folder);
void refine_mesh_segmentation(const char* _machine, const char* _train_folder);
}