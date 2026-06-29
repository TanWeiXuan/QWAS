cmake_minimum_required(VERSION 3.16)

project(QWAS LANGUAGES CXX)

option(QWAS_BUILD_NATIVE "Build the desktop game" ON)
option(QWAS_BUILD_WEB "Build the Emscripten web game" OFF)

set(QWAS_IS_EMSCRIPTEN FALSE)
if(EMSCRIPTEN OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(QWAS_IS_EMSCRIPTEN TRUE)
endif()

if(QWAS_BUILD_WEB AND NOT QWAS_IS_EMSCRIPTEN)
    message(FATAL_ERROR "QWAS_BUILD_WEB requires configuring with Emscripten. Use `emcmake cmake ... -DQWAS_BUILD_WEB=ON`.")
endif()

if(QWAS_IS_EMSCRIPTEN AND QWAS_BUILD_NATIVE)
    message(FATAL_ERROR "QWAS_BUILD_NATIVE cannot be built with Emscripten. Configure web builds with -DQWAS_BUILD_NATIVE=OFF.")
endif()

# CMake 4.x dropped support for cmake_minimum_required < 3.5.
# This allows FetchContent'd projects (like raylib 5.0) to still configure.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE INTERNAL "")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

if(QWAS_BUILD_WEB)
    set(PLATFORM Web CACHE STRING "raylib target platform" FORCE)
endif()

FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        6.0
    GIT_SHALLOW    TRUE
)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(raylib)

add_library(qwas_game
    src/drone.cpp
    src/game.cpp
    src/qwas_app.cpp
)

target_include_directories(qwas_game PUBLIC include)
target_link_libraries(qwas_game PUBLIC raylib)

if(QWAS_BUILD_NATIVE)
    add_executable(QWAS
        src/main.cpp
    )

    target_link_libraries(QWAS PRIVATE qwas_game)

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
endif()

if(QWAS_BUILD_WEB)
    set(QWAS_WEB_DIST_DIR "${CMAKE_BINARY_DIR}/dist")

    add_executable(qwas_web
        src/web/web_main.cpp
    )

    target_link_libraries(qwas_web PRIVATE qwas_game)
    set_target_properties(qwas_web PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${QWAS_WEB_DIST_DIR}"
        OUTPUT_NAME "index"
        SUFFIX ".html"
    )

    target_link_options(qwas_web PRIVATE
        "-sUSE_GLFW=3"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sGL_ENABLE_GET_PROC_ADDRESS=1"
        "--shell-file=${CMAKE_CURRENT_LIST_DIR}/web/emscripten_shell.html"
    )

    add_custom_command(TARGET qwas_web POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_LIST_DIR}/doc/qwas.png"
            "${QWAS_WEB_DIST_DIR}/qwas.png"
        COMMENT "Copying social preview image"
    )
endif()
