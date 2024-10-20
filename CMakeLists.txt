cmake_minimum_required(VERSION 3.22.1) # my version

set(COM_ARGS)
set(LINK_LIBS)
set(SRC_FILES src/__alc.cpp src/stdalgo.cpp)

# --- Options --- #
  option(NV "Compile for Nvidia Graphic Cards" OFF)
  option(AMD "Compile for AMD Graphic Cards (NOT_IMPLEMENTED)" OFF)
# --- END(Options) --- #
# --- Parse Options --- #
  if (NV)
    Project(Alacrity VERSION 1.0.0 LANGUAGES CXX CUDA)
    find_package(CUDA REQUIRED)
    list(APPEND LINK_LIBS ${CUDA_LIBRARIES})
    include_directories(${CUDA_INCLUDE_DIRS})
    # list(APPEND SRC_FILES)
  elseif (AMD)
    message( FATAL_ERROR "NOT_IMPLEMENTED: AMD")
    # list(APPEND SRC_FILES)
  else()
    Project(Alacrity VERSION 1.0.0 LANGUAGES CXX)
  endif()
# --- END(Parse Options) --- #

include(GNUInstallDirs)

add_library(Alacrity SHARED ${SRC_FILES})
add_library(Alacrity::Alacrity ALIAS Alacrity)

target_link_libraries(Alacrity PRIVATE ${LINK_LIBS})
target_compile_definitions(Alacrity PUBLIC ${COM_ARGS})

target_include_directories(
  Alacrity
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS Alacrity
  EXPORT AlacrityTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Include the CMakePackageConfigHelpers module
include(CMakePackageConfigHelpers)

# Set some properties for your library target
set_property(TARGET Alacrity PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET Alacrity PROPERTY SOVERSION 1)
set_property(TARGET Alacrity PROPERTY
  INTERFACE_Alacrity_MAJOR_VERSION 1)
set_property(TARGET Alacrity APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING Alacrity_MAJOR_VERSION
)

# Generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/AlacrityConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

# Create the config file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AlacrityConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/AlacrityConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Alacrity
)

# Install the config and version files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/AlacrityConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/AlacrityConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Alacrity
)

# Export the targets for the build tree
export(EXPORT AlacrityTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/AlacrityTargets.cmake"
  NAMESPACE Alacrity::
)

# Install the export set for the install tree
install(EXPORT AlacrityTargets
  FILE AlacrityTargets.cmake
  NAMESPACE Alacrity::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Alacrity
)


