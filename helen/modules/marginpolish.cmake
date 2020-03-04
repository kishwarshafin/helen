# build htslib
set(marginpolish_PREFIX ${CMAKE_BINARY_DIR}/marginpolish)
set (FLAGS "-fPIC")
# Enable ExternalProject CMake module
include(ExternalProject)

ExternalProject_Add(marginpolish
        GIT_REPOSITORY https://github.com/UCSC-nanopore-cgl/MarginPolish.git
        GIT_TAG tags/v1.3.0
        PREFIX ${marginpolish_PREFIX}
        BUILD_COMMAND cmake . && make CFLAGS=${CMAKE_C_FLAGS}
        INSTALL_COMMAND ""
        )

ExternalProject_Get_Property(marginpolish BINARY_DIR)

# MARGINPOLISH SOURCES
set(MARGINPOLISH_BINARY_DIR ${BINARY_DIR})