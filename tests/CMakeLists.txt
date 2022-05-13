cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(h5demos CXX C CUDA)
set(CMAKE_C_FLAGS $ENV{CFLAGS})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#find_package(MPI REQUIRED)
find_package(ROOT REQUIRED COMPONENTS)
find_package(HDF5 REQUIRED)
#find_package(TBB REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

message("STATUS OF HDF5_FOUND IS ${HDF5_FOUND}")
#set(HDF5_PREFER_PARALLEL TRUE)

set(CUDA_INCLUDE_DIRS "/usr/common/software/sles15_cgpu/cuda/11.1.1/include/")

set(MPI_INCLUDE_DIRS "/usr/common/software/mpich/3.3.1-debug/gcc/7.3.0/include")
set(MPI_LIBRARIES "/usr/common/software/mpich/3.3.1-debug/gcc/7.3.0/lib/libmpi.so")

set(TBB_LIBRARIES "/usr/common/software/intel/parallel_studio_xe_2020_update3_cluster_edition/tbb/lib/intel64/gcc4.8/libtbb.so")

set(TBB_INCLUDE_DIRS "/usr/common/software/intel/parallel_studio_xe_2020_update3_cluster_edition/tbb/include")

message(STATUS "C++ compiler is : ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ compiler identification is : ${CMAKE_CXX_COMPILER_ID}")

include_directories (SYSTEM 

		    #${MPI_CXX_INCLUDE_DIRS}
		    #${MPI_C_INCLUDE_DIRS}
            ${MPI_INCLUDE_DIRS}
		    ${HDF5_INCLUDE_DIRS}
            ${TBB_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
		    ${ROOT_INCLUDE_DIRS}
            )
		    
#link_directories ($ENV{ROOTSYS}/lib)

set (libraries ${libraries}
               ${ROOT_LIBRARIES}
	       #${MPI_CXX_LIBRARIES}
	       #${MPI_C_LIBRARIES}
           #${MPI_LIBRARIES}
           ${TBB_LIBRARIES}
	       ${HDF5_LIBRARIES}
              ROOT::Core
              ROOT::RIO
              ROOT::Tree
              ROOT::Hist)


add_executable(getdunehdf5 
               GenerateDataInHDF5.cc    
               getdunehdf5.cc)
target_link_libraries(getdunehdf5 ${libraries})

add_executable(readdunehdf5 
               ReadRawDataHDF5.cc    
               readdunehdf5.cc)
target_link_libraries(readdunehdf5 ${libraries})

add_executable(test_4vec test_4vec.cu)
target_link_libraries(test_4vec ${libraries})

#test if the scheduler is working...
add_executable(test_offloader
               KernelStatus.cc
               TaskArena.cc
               OffloadIntoHost.cc
               OffloadKernelTasks.cc
               ReadRawDataHDF5.cc
               test_offloader.cu)
               
target_link_libraries(test_offloader ${libraries})