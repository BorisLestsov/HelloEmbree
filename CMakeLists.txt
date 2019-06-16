cmake_minimum_required(VERSION 3.5)
project(HelloEmbree)

# Setting Intel Thread Building Blocks
#set(TBB_INSTALL_DIR         "/home/denispavlov/Software/tbb-2019_U2")
#set(TBB_INCLUDE             "$TBB_INSTALL_DIR/include")
#set(TBB_LIBRARY_RELEASE     "${TBB_INSTALL_DIR}/build/linux_intel64_gcc_cc7_libc2.27_kernel4.15.0_release")
#set(TBB_LIBRARY_DEBUG       "${TBB_INSTALL_DIR}/build/linux_intel64_gcc_cc7_libc2.27_kernel4.15.0_debug")

# Setting Intel Embree
set(EMBREE_INSTALL_DIR      "/home/b.lestsov/rend/HelloEmbree/embree-3.5.2.x86_64.linux")
set(EMBREE_INCLUDE          "${EMBREE_INSTALL_DIR}/include")
set(EMBREE_LIBRARY          "${EMBREE_INSTALL_DIR}/lib")

include_directories(${TBB_INCLUDE} ${EMBREE_INCLUDE})
link_directories(${TBB_LIBRARY_RELEASE} ${EMBREE_LIBRARY})


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(CORE_SOURCE )

set(SOURCE_FILES main.cpp
        vector3.h
        ray.h
        bxdf.h
        primitive.h
        utils.h)

add_executable(HelloEmbree ${SOURCE_FILES} ${CORE_SOURCE})
target_link_libraries(HelloEmbree embree3)
#target_link_libraries(HelloEmbree tbb embree3)
