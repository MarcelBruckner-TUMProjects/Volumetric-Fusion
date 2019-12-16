// Here just to compile the needed classes from the common_utils library.
#include <vector_types.h>
#include <vector_functions.hpp>

#include <common_utils/IncludeImplCUDA.hpp>
#include <common_utils/memory_managment/MemoryContainerCUDA.cuh>
#include <matrix_lib/matrix_structures/MatrixStructuresInclude.h>

// Data structures, used in some tests.
template class common_utils::MemoryContainer<float4>;
template class common_utils::MemoryContainer<int4>;
template class common_utils::MemoryContainer<matrix_lib::Mat4f>;
template class common_utils::MemoryContainer<int>;
template class common_utils::MemoryContainer<unsigned>;
template class common_utils::MemoryContainer<bool>;
