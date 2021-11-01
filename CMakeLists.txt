cmake_minimum_required(VERSION 3.17)

set(SPIDRPLUGIN "SpidrPlugin")

PROJECT(${SPIDRPLUGIN})

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
    src/SpidrAnalysisQtWrapper.h
    src/SpidrAnalysisQtWrapper.cpp
    src/TsneComputationQtWrapper.h
    src/TsneComputationQtWrapper.cpp
    src/SpidrData.h
    src/SpidrOptions.h

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

set(ACTIONS
    src/GeneralSpidrSettingsAction.h
    src/GeneralSpidrSettingsAction.cpp
    src/SpidrSettingsAction.h
    src/SpidrSettingsAction.cpp
    src/TsneComputationAction.h
    src/TsneComputationAction.cpp
    src/AdvancedTsneSettingsAction.h
    src/AdvancedTsneSettingsAction.cpp
    src/DimensionSelectionAction.h
    src/DimensionSelectionAction.cpp
)


file(GLOB WIDGETS src/*Widget.*)
file(GLOB UIS ui/*.ui)
file(GLOB JSON src/*.json)

set(SOURCES ${PLUGIN} ${ACTIONS} ${DIMENSIONSELECTIONHOLDER} ${WIDGETS})
set(AUX ${JSON})

source_group(Plugin FILES ${PLUGIN})
source_group(Actions FILES ${ACTIONS})
source_group(Spidrlib FILES ${SPIDRLIB})
source_group(Widgets FILES ${WIDGETS})
source_group(DimensionSelectionHolder FILES ${DIMENSIONSELECTIONHOLDER})
source_group(UI FILES ${UIS})
source_group(Aux FILES ${AUX})

QT5_WRAP_UI(UI_HEADERS ${UIS})

add_library(${SPIDRPLUGIN} SHARED ${SOURCES} ${SPIDRLIB} ${UIS} ${AUX})

target_include_directories(${SPIDRPLUGIN} PRIVATE "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/include/")
target_include_directories(${SPIDRPLUGIN} PRIVATE "external/Spidr/include/")

target_compile_definitions(${SPIDRPLUGIN} PRIVATE QT_MESSAGELOGCONTEXT)

target_compile_features(${SPIDRPLUGIN} PUBLIC cxx_std_17)

target_link_libraries(${SPIDRPLUGIN} PRIVATE Qt5::Widgets)
target_link_libraries(${SPIDRPLUGIN} PRIVATE Qt5::WebEngineWidgets)
target_link_libraries(${SPIDRPLUGIN} PRIVATE "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/lib/HDPS_Public.lib")
target_link_libraries(${SPIDRPLUGIN} PRIVATE "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/lib/PointData.lib")

add_subdirectory("external/Spidr")
target_link_libraries(${SPIDRPLUGIN} PRIVATE SpidrLib)

# Prevent glfw including the OpenGL headers
# On Windows and Linux these come from glad in the HDILib, which is part of the SpidrLib
add_definitions(-DGLFW_INCLUDE_NONE)

add_custom_command(TARGET ${SPIDRPLUGIN} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "$<TARGET_FILE:${SPIDRPLUGIN}>"
    "$ENV{HDPS_INSTALL_DIR}/$<CONFIGURATION>/Plugins/$<TARGET_FILE_NAME:${SPIDRPLUGIN}>"
)