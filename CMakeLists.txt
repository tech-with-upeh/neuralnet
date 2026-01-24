cmake_minimum_required(VERSION 3.16)

project(Neuraln LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)


add_executable(Neuraln
  src/test.cpp
)

target_include_directories(Neuraln
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)


# Optional: warnings (safe)
if (MSVC)
    target_compile_options(Neuraln PRIVATE /w)  #W4
else()
    target_compile_options(Neuraln PRIVATE -w)
endif()