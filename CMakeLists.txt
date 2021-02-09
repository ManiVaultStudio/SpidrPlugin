cmake_minimum_required(VERSION 3.1.0)

set(PROJECT "SpidrPlugin")

PROJECT(${PROJECT})

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32 /EHsc /MP")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

find_package(Qt5 COMPONENTS Widgets WebEngineWidgets REQUIRED)
# use header-only libs, thus no components specified
find_package(Boost REQUIRED) 
find_package(Eigen3 3.3 REQUIRED)

set(PLUGIN
    src/SpidrPlugin.h
    src/SpidrPlugin.cpp
    src/SpidrPlugin.json
    src/SpidrAnalysis.h
    src/SpidrAnalysis.cpp
    src/TsneComputation.h
    src/TsneComputation.cpp
)

set(DIMENSIONSELECTIONHOLDER
    src/ModelResetter.h
    src/ModelResetter.cpp
    src/DimensionSelectionHolder.h
    src/DimensionSelectionHolder.cpp
    src/DimensionSelectionItemModel.h
    src/DimensionSelectionItemModel.cpp
    src/DimensionSelectionProxyModel.h
    src/DimensionSelectionProxyModel.cpp
)

file(GLOB WIDGETS src/*Widget.*)
file(GLOB UIS ui/*.ui)
file(GLOB JSON src/*.json)
file(GLOB SPIDRLIB lib/Spidr/src/*)

set(SOURCES ${PLUGIN} ${DIMENSIONSELECTIONHOLDER} ${WIDGETS})
set(AUX ${JSON})

source_group(Plugin FILES ${PLUGIN})
source_group(Spidrlib FILES ${SPIDRLIB})
source_group(Widgets FILES ${WIDGETS})
source_group(DimensionSelectionHolder FILES ${DIMENSIONSELECTIONHOLDER})
source_group(UI FILES ${UIS})
source_group(Aux FILES ${AUX})

QT5_WRAP_UI(UI_HEADERS ${UIS})

add_library(${PROJECT} SHARED ${SOURCES} ${SPIDRLIB} ${UIS} ${AUX})

include_directories("$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/include/")
include_directories ("lib/HDI/include")
include_directories ("lib/hnswlib/")
include_directories ("lib/Spidr/src/")
include_directories ("lib/Spidr/external/spdlog/include/")
include_directories(${Boost_INCLUDE_DIRS})

if(MSVC)
    include_directories ("lib/Flann/Win/include")
	ADD_DEFINITIONS(/arch:AVX)
	add_definitions(-DUSE_SSE)
	add_definitions(-DUSE_AVX)
endif(MSVC)

if(APPLE)
    include_directories ("lib/Flann/OSX/include")
endif(APPLE)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories ("lib/Flann/Linux/include")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

target_compile_definitions(${PROJECT} PRIVATE QT_MESSAGELOGCONTEXT)

# Request C++17, in order to use std::for_each_n with std::execution::par_unseq.
set_property(TARGET ${PROJECT} PROPERTY CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
	message (STATUS "OpenMP found ${OpenMP_CXX_LIB_NAMES}")
	target_link_libraries(${PROJECT} OpenMP::OpenMP_CXX)
else()
    message(WARNING, "OpenMP not found!")    
endif()

target_link_libraries(${PROJECT} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT} Eigen3::Eigen)
target_link_libraries(${PROJECT} Qt5::Widgets)
target_link_libraries(${PROJECT} Qt5::WebEngineWidgets)
target_link_libraries(${PROJECT} "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/lib/HDPS_Public.lib")
target_link_libraries(${PROJECT} "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/lib/PointData.lib")

if(MSVC)
    MESSAGE( STATUS "Linking Windows libraries...")
    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/lib/HDI/Win/Debug/hdidimensionalityreduction.lib")
    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/lib/HDI/Win/Debug/hdidata.lib")
    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/lib/HDI/Win/Debug/hdiutils.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/lib/HDI/Win/Release/hdidimensionalityreduction.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/lib/HDI/Win/Release/hdidata.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/lib/HDI/Win/Release/hdiutils.lib")

    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/lib/Flann/Win/Debug/flann_cpp_s.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/lib/Flann/Win/Release/flann_cpp_s.lib")
endif(MSVC)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    MESSAGE( STATUS "Linking Linux libraries...")
    target_link_libraries(${PROJECT} "${CMAKE_CURRENT_SOURCE_DIR}/lib/AtSNE/Linux/libbh_t_sne_library.a")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

if(APPLE)
    MESSAGE( STATUS "Linking Mac OS X libraries...")

    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/lib/AtSNE/OSX/Debug/libbh_t_sne_library.a")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/lib/AtSNE/OSX/Release/libbh_t_sne_library.a")

    target_link_libraries(${PROJECT} "${CMAKE_CURRENT_SOURCE_DIR}/lib/Flann/OSX/libflann_s.a")
endif(APPLE)

add_custom_command(TARGET ${PROJECT} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "$<TARGET_FILE:${PROJECT}>"
    "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/Plugins/$<TARGET_FILE_NAME:${PROJECT}>"
)