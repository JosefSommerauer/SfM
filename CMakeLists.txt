cmake_minimum_required(VERSION 2.8)
project( SFM )
set (CMAKE_CXX_STANDARD 11)
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package( OpenCV REQUIRED )
find_package(PCL 1.3 REQUIRED COMPONENTS common io visualization filters)
find_package(CUDA QUIET REQUIRED)

include_directories(./include)
include_directories(./include/cuda)

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

#IF(CUDA_FOUND)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)
    cuda_compile(trian_cuda triangulation.cu)
    cuda_add_library(trian_cuda triangulation.cu)
    add_definitions(-DGPU_OPENCV_ENABLE)
#ENDIF()

#add_library(CUDA STATIC ${CUDA_LIBRARY_DIRS})

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# pcl
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



file(GLOB SFM_SRC
    "*.h"
    "*.cpp"
)

add_executable( SFM ${SFM_SRC})
target_link_libraries( SFM trian_cuda ${OpenCV_LIBS} 
	${PCL_COMMON_LIBRARIES}  
	${PCL_IO_LIBRARIES} 
	${PCL_VISUALIZATION_LIBRARIES} 
	${PCL_FILTERS_LIBRARIES} 
	${PCL_IO_LIBRARIES} 
	${CUDA_LIBRARIES} 
	cudart cublas 
	)
