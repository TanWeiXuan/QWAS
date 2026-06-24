cmake_minimum_required(VERSION 3.16)
project(QWAS LANGUAGES CXX)

# CMake 4.x dropped support for cmake_minimum_required < 3.5.
# This allows FetchContent'd projects (like raylib 5.0) to still configure.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE INTERNAL "")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        6.0
    GIT_SHALLOW    TRUE
)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

add_executable(QWAS
    src/main.cpp
    src/drone.cpp
    src/game.cpp
)

target_include_directories(QWAS PRIVATE include)
target_link_libraries(QWAS PRIVATE raylib)

if(APPLE)
    target_link_libraries(QWAS PRIVATE
        "-framework Cocoa"
        "-framework IOKit"
        "-framework OpenGL"
    )
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(QWAS PRIVATE dl pthread)
endif()
