cmake_minimum_required(VERSION 3.21)

option(USE_AVX "Use AVX instructions sets if available" ON)
option(USE_OPENMP "Use OpenMP - by default ON" ON)

set(SPIDRPLUGIN "SpidrPlugin")
PROJECT(${SPIDRPLUGIN} CXX)

# -----------------------------------------------------------------------------
# CMake settings
# -----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_AUTOMOC ON)

# Check if the directory to the ManiVault installation has been provided
if(NOT DEFINED MV_INSTALL_DIR)
    set(MV_INSTALL_DIR "" CACHE PATH "Directory where ManiVault is installed")
    message(FATAL_ERROR "Please set MV_INSTALL_DIR to the directory where ManiVault is installed")
endif()
file(TO_CMAKE_PATH ${MV_INSTALL_DIR} MV_INSTALL_DIR)

message(STATUS "Install to ${MV_INSTALL_DIR}")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32 /EHsc /MP /W3 /permissive- /Zc:__cplusplus")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

# -----------------------------------------------------------------------------
# Library dependencies
# -----------------------------------------------------------------------------
find_package(Qt6 COMPONENTS Widgets WebEngineWidgets REQUIRED)
find_package(OpenMP)

set(CREATE_EXAMPLE OFF CACHE BOOL "Creates an example executable using the Spidr library")
set(CREATE_PYTHON_WRAPPER OFF CACHE BOOL "Creates a python wrapper of the Spidr library")
set(USE_MACOSSDK_OPENGL OFF CACHE BOOL "Use MACOS SDK provided glad.h instead of hdi/utils/glad/glad.h")

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

set(PLUGIN_MOC_HEADERS
    src/SpidrPlugin.h
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

qt_wrap_cpp(SPIDRPLUGIN_MOC ${PLUGIN_MOC_HEADERS} TARGET ${SPIDRPLUGIN})
target_sources(${SPIDRPLUGIN} PRIVATE ${SPIDRPLUGIN_MOC})

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------
target_compile_features(${SPIDRPLUGIN} PRIVATE cxx_std_17)

# Hardware accelations: AVX
check_and_link_AVX(${SPIDRPLUGIN} ${USE_AVX})

if(MSVC)
    message(STATUS "Prevent compilation error on MSVC: Use AVX instead of AVX2")
    get_target_property(SPIDRPLUGIN_COMPILE_OPTIONS ${SPIDRPLUGIN} COMPILE_OPTIONS)
    list(REMOVE_ITEM SPIDRPLUGIN_COMPILE_OPTIONS "/arch:AVX2")
    list(APPEND SPIDRPLUGIN_COMPILE_OPTIONS "/arch:AVX")
    set_target_properties(${SPIDRPLUGIN} PROPERTIES COMPILE_OPTIONS "${SPIDRPLUGIN_COMPILE_OPTIONS}")
endif()

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
target_include_directories(${SPIDRPLUGIN} PRIVATE "${MV_INSTALL_DIR}/$<CONFIGURATION>/include/")
target_include_directories(${SPIDRPLUGIN} PRIVATE "external/Spidr/include/")

#------------------------------------------------------------------------------
# Compile defines
#------------------------------------------------------------------------------
if(USE_MACOSSDK_OPENGL)
    target_compile_definitions(${SPIDRPLUGIN} PRIVATE USE_MACOSSDK_OPENGL)
endif()

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
# link against Qt
target_link_libraries(${SPIDRPLUGIN} PRIVATE Qt6::Widgets)
target_link_libraries(${SPIDRPLUGIN} PRIVATE Qt6::WebEngineWidgets)

# Link against ManiVault core and data plugins
set(MV_LINK_PATH "${MV_INSTALL_DIR}/$<CONFIGURATION>/lib")
set(PLUGIN_LINK_PATH "${MV_INSTALL_DIR}/$<CONFIGURATION>/$<IF:$<CXX_COMPILER_ID:MSVC>,lib,Plugins>")
set(MV_LINK_SUFFIX $<IF:$<CXX_COMPILER_ID:MSVC>,${CMAKE_LINK_LIBRARY_SUFFIX},${CMAKE_SHARED_LIBRARY_SUFFIX}>)

set(MV_LINK_LIBRARY "${MV_LINK_PATH}/${CMAKE_SHARED_LIBRARY_PREFIX}MV_Public${MV_LINK_SUFFIX}")
set(POINTDATA_LINK_LIBRARY "${PLUGIN_LINK_PATH}/${CMAKE_SHARED_LIBRARY_PREFIX}PointData${MV_LINK_SUFFIX}") 
set(IMAGEDATA_LINK_LIBRARY "${PLUGIN_LINK_PATH}/${CMAKE_SHARED_LIBRARY_PREFIX}ImageData${MV_LINK_SUFFIX}") 

target_link_libraries(${SPIDRPLUGIN} PRIVATE "${MV_LINK_LIBRARY}")
target_link_libraries(${SPIDRPLUGIN} PRIVATE "${POINTDATA_LINK_LIBRARY}")
target_link_libraries(${SPIDRPLUGIN} PRIVATE "${IMAGEDATA_LINK_LIBRARY}")

# Link against SpidrLib
target_link_libraries(${SPIDRPLUGIN} PRIVATE SpidrLib)

if(USE_OPENMP AND OpenMP_CXX_FOUND)
	target_link_libraries(${SPIDRPLUGIN} PRIVATE OpenMP::OpenMP_CXX)
endif()

# -----------------------------------------------------------------------------
# Target installation
# -----------------------------------------------------------------------------
install(TARGETS ${SPIDRPLUGIN}
    RUNTIME DESTINATION Plugins COMPONENT PLUGINS # Windows .dll
    LIBRARY DESTINATION Plugins COMPONENT PLUGINS # Linux/Mac .so
)

install(FILES $<TARGET_FILE:glfw> DESTINATION .)

add_custom_command(TARGET ${SPIDRPLUGIN} POST_BUILD
	COMMAND "${CMAKE_COMMAND}"
	--install ${CMAKE_CURRENT_BINARY_DIR}
	--config $<CONFIGURATION>
	--prefix ${MV_INSTALL_DIR}/$<CONFIGURATION>
)

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
if(MSVC)
	# Automatically set the debug environment (command + working directory) for MSVC
	set_property(TARGET ${SPIDRPLUGIN} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<IF:$<CONFIG:DEBUG>,${MV_INSTALL_DIR}/Debug,${MV_INSTALL_DIR}/Release>)
	set_property(TARGET ${SPIDRPLUGIN} PROPERTY VS_DEBUGGER_COMMAND $<IF:$<CONFIG:DEBUG>,"${MV_INSTALL_DIR}/debug/ManiVault Studio.exe","${MV_INSTALL_DIR}/release/ManiVault Studio.exe">) 
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${SPIDRPLUGIN})
endif()
