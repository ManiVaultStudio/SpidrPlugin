cmake_minimum_required(VERSION 3.21)

# setup vcpkg cmake toolchain
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CRT_LINKAGE dynamic)
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/external/Spidr/external/vcpkg/scripts/buildsystems/vcpkg.cmake"
  CACHE STRING "Vcpkg toolchain file")

set(SPIDRPLUGIN "SpidrPlugin")

PROJECT(${SPIDRPLUGIN})

# -----------------------------------------------------------------------------
# CMake settings
# -----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(INSTALL_DIR $ENV{HDPS_INSTALL_DIR})
message(STATUS "Install to ${INSTALL_DIR}")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32 /EHsc /MP /permissive- /Zc:__cplusplus")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

# -----------------------------------------------------------------------------
# Library dependencies
# -----------------------------------------------------------------------------
find_package(Qt6 COMPONENTS Widgets WebEngineWidgets REQUIRED)

if(MSVC)
	set(USE_HDILIBSLIM ON CACHE BOOL "Use the slim HDILib version")
endif()
set(CREATE_EXAMPLE OFF CACHE BOOL "Creates an example executable using the Spidr library")
set(CREATE_PYTHON_WRAPPER OFF CACHE BOOL "Creates a python wrapper of the Spidr library")

add_subdirectory("external/Spidr")

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
set(PLUGIN
    src/SpidrPlugin.h
    src/SpidrPlugin.cpp
    src/SpidrAnalysisQtWrapper.h
    src/SpidrAnalysisQtWrapper.cpp
    src/TsneComputationQtWrapper.h
    src/TsneComputationQtWrapper.cpp
    src/OffscreenBuffer.h
    src/OffscreenBuffer.cpp
)

set(ACTIONS
    src/GeneralSpidrSettingsAction.h
    src/GeneralSpidrSettingsAction.cpp
    src/SpidrSettingsAction.h
    src/SpidrSettingsAction.cpp
    src/SpidrComputationAction.h
    src/SpidrComputationAction.cpp
    src/HistBinAction.h
    src/HistBinAction.cpp
    src/AdvancedTsneSettingsAction.h
    src/AdvancedTsneSettingsAction.cpp
    src/DimensionSelectionAction.h
    src/DimensionSelectionAction.cpp
    src/BackgroundSelectionAction.h
    src/BackgroundSelectionAction.cpp
)

set(AUX
    src/SpidrPlugin.json
)

set(SOURCES ${PLUGIN} ${ACTIONS} ${AUX})

source_group(Plugin FILES ${PLUGIN})
source_group(Actions FILES ${ACTIONS})
source_group(Aux FILES ${AUX})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------
add_library(${SPIDRPLUGIN} SHARED ${SOURCES})

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------
target_compile_features(${SPIDRPLUGIN} PRIVATE cxx_std_17)

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
target_include_directories(${SPIDRPLUGIN} PRIVATE "${INSTALL_DIR}/$<CONFIGURATION>/include/")
target_include_directories(${SPIDRPLUGIN} PRIVATE "external/Spidr/include/")

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
if(MSVC)
    set(LIB_SUFFIX "${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
    set(LIB_SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

target_link_libraries(${SPIDRPLUGIN} PRIVATE Qt6::Widgets)
target_link_libraries(${SPIDRPLUGIN} PRIVATE Qt6::WebEngineWidgets)
target_link_libraries(${SPIDRPLUGIN} PRIVATE "${INSTALL_DIR}/$<CONFIGURATION>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}HDPS_Public${LIB_SUFFIX}")
target_link_libraries(${SPIDRPLUGIN} PRIVATE "${INSTALL_DIR}/$<CONFIGURATION>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}PointData${LIB_SUFFIX}")
target_link_libraries(${SPIDRPLUGIN} PRIVATE "${INSTALL_DIR}/$<CONFIGURATION>/lib/${CMAKE_SHARED_LIBRARY_PREFIX}ImageData${LIB_SUFFIX}")

target_link_libraries(${SPIDRPLUGIN} PRIVATE SpidrLib)

# -----------------------------------------------------------------------------
# Target installation
# -----------------------------------------------------------------------------
add_custom_command(TARGET ${SPIDRPLUGIN} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "$<TARGET_FILE:${SPIDRPLUGIN}>"
    "${INSTALL_DIR}/$<CONFIGURATION>/Plugins/$<TARGET_FILE_NAME:${SPIDRPLUGIN}>"
)

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
# Automatically set the debug environment (command + working directory) for MSVC in debug mode
set_property(TARGET ${SPIDRPLUGIN} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<IF:$<CONFIG:DEBUG>,${INSTALL_DIR}/Debug,${INSTALL_DIR}/Release>)
set_property(TARGET ${SPIDRPLUGIN} PROPERTY VS_DEBUGGER_COMMAND $<IF:$<CONFIG:DEBUG>,${INSTALL_DIR}/Debug/HDPS.exe,${INSTALL_DIR}/Release/HDPS.exe>) 