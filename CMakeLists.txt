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
find_package(OpenMP REQUIRED)

if(UNIX AND NOT APPLE)
   find_package(TBB REQUIRED)
endif()

find_package(ManiVault COMPONENTS Core PointData ImageData CONFIG)

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
target_include_directories(${SPIDRPLUGIN} PRIVATE "${ManiVault_INCLUDE_DIR}")
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

target_link_libraries(${SPIDRPLUGIN} PRIVATE ManiVault::Core)
target_link_libraries(${SPIDRPLUGIN} PRIVATE ManiVault::PointData)
target_link_libraries(${SPIDRPLUGIN} PRIVATE ManiVault::ImageData)

# Link against SpidrLib
target_link_libraries(${SPIDRPLUGIN} PRIVATE SpidrLib)

target_link_libraries(${SPIDRPLUGIN} PRIVATE OpenMP::OpenMP_CXX)
if(UNIX AND NOT APPLE)
   target_link_libraries(${SPIDRPLUGIN} PRIVATE TBB::tbb)
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
	--prefix ${ManiVault_INSTALL_DIR}/$<CONFIGURATION>
)

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
if(MSVC)
	# Automatically set the debug environment (command + working directory) for MSVC
	set_property(TARGET ${SPIDRPLUGIN} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<IF:$<CONFIG:DEBUG>,${ManiVault_INSTALL_DIR}/Debug,${ManiVault_INSTALL_DIR}/Release>)
	set_property(TARGET ${SPIDRPLUGIN} PROPERTY VS_DEBUGGER_COMMAND $<IF:$<CONFIG:DEBUG>,"${ManiVault_INSTALL_DIR}/debug/ManiVault Studio.exe","${ManiVault_INSTALL_DIR}/release/ManiVault Studio.exe">) 
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${SPIDRPLUGIN})
endif()
