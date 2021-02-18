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
include_directories("external/Spidr/external/HDILib/HDI/include")
include_directories("external/Spidr/external/hnswlib/")
include_directories("external/Spidr/include/")

if(MSVC)
    include_directories ("external/Spidr/external/HDILib/Flann/Win/include")
	ADD_DEFINITIONS(/arch:AVX)
	add_definitions(-DUSE_SSE)
	add_definitions(-DUSE_AVX)
endif(MSVC)

target_compile_definitions(${PROJECT} PRIVATE QT_MESSAGELOGCONTEXT)

# Request C++17, in order to use std::for_each_n with std::execution::par_unseq.
set_property(TARGET ${PROJECT} PROPERTY CXX_STANDARD 17)

target_link_libraries(${PROJECT} Qt5::Widgets)
target_link_libraries(${PROJECT} Qt5::WebEngineWidgets)
target_link_libraries(${PROJECT} "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/lib/HDPS_Public.lib")
target_link_libraries(${PROJECT} "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/lib/PointData.lib")

add_subdirectory("external/Spidr")
target_link_libraries(${PROJECT} Spidr)

if(MSVC)
    MESSAGE( STATUS "Linking Windows libraries...")
    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/external/Spidr/external/HDILib/HDI/Win/Debug/hdidimensionalityreduction.lib")
    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/external/Spidr/external/HDILib/HDI/Win/Debug/hdidata.lib")
    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/external/Spidr/external/HDILib/HDI/Win/Debug/hdiutils.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/Spidr//external/HDILib/HDI/Win/Release/hdidimensionalityreduction.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/Spidr/external/HDILib/HDI/Win/Release/hdidata.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/Spidr/external/HDILib/HDI/Win/Release/hdiutils.lib")

    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/external/Spidr/external/HDILib/Flann/Win/Debug/flann_cpp_s.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/Spidr/external/HDILib/Flann/Win/Release/flann_cpp_s.lib")
else()
	message(ERROR, "Only windows tested")    
endif(MSVC)

add_custom_command(TARGET ${PROJECT} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "$<TARGET_FILE:${PROJECT}>"
    "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/Plugins/$<TARGET_FILE_NAME:${PROJECT}>"
)