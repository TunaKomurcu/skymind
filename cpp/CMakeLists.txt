cmake_minimum_required(VERSION 3.20)
project(skymind_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
  add_compile_options(/utf-8 /W4)
else()
  add_compile_options(-Wall -Wextra)
endif()

option(SKYMIND_BUILD_SERVER "llama.cpp tabanli sunucuyu derle" ON)
option(SKYMIND_BUILD_TESTS "GoogleTest testlerini derle" ON)
set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp kaynak dizini (bos: FetchContent ile indirilir)")
set(LLAMA_CPP_TAG "b12655" CACHE STRING "FetchContent icin llama.cpp etiketi")
set(SKYMIND_LLAMA_PREBUILT_DIR "" CACHE PATH "Hazir llama.dll/llama.lib dizini (bos: kaynaktan derle)")
option(SKYMIND_WITH_CUDA "Hazir CUDA arka ucunu (ggml-cuda.dll, cudart, cublas) da kopyala" OFF)

include(FetchContent)

# --- nlohmann/json: llama.cpp'nin vendor kopyasi varsa onu, yoksa resmi surumu kullan ----------
if(SKYMIND_BUILD_SERVER)
  if(LLAMA_CPP_DIR STREQUAL "")
    FetchContent_Declare(llama_cpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG ${LLAMA_CPP_TAG}
      GIT_SHALLOW TRUE)
    FetchContent_Populate(llama_cpp)
    set(LLAMA_CPP_DIR ${llama_cpp_SOURCE_DIR})
  endif()
  set(JSON_INCLUDE_DIR ${LLAMA_CPP_DIR}/vendor)
else()
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# --- cekirdek kutuphane: dogrulama, istem, metrikler, servis mantigi (llama.cpp'den bagimsiz) ---
add_library(skymind_core STATIC
  src/validate.cpp src/prompt.cpp src/metrics.cpp src/service.cpp)
target_include_directories(skymind_core PUBLIC include)
if(SKYMIND_BUILD_SERVER)
  target_include_directories(skymind_core PUBLIC ${JSON_INCLUDE_DIR})
else()
  target_link_libraries(skymind_core PUBLIC nlohmann_json::nlohmann_json)
endif()

# --- sunucu ------------------------------------------------------------------------------------
if(SKYMIND_BUILD_SERVER)
  if(SKYMIND_LLAMA_PREBUILT_DIR)
    # Hazir llama.dll (+ ggml*.dll, istege bagli CUDA arka ucu) kullan; basliklar LLAMA_CPP_DIR'den.
    # Ayni etiketten (LLAMA_CPP_TAG) olmalidir. llama.lib: scripts/make_import_lib.py ile uretilir.
    add_library(llama SHARED IMPORTED GLOBAL)
    set_target_properties(llama PROPERTIES
      IMPORTED_LOCATION ${SKYMIND_LLAMA_PREBUILT_DIR}/llama.dll
      IMPORTED_IMPLIB ${SKYMIND_LLAMA_PREBUILT_DIR}/llama.lib
      INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_CPP_DIR}/include;${LLAMA_CPP_DIR}/ggml/include")
  else()
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)
  endif()

  add_library(skymind_httplib STATIC ${LLAMA_CPP_DIR}/vendor/cpp-httplib/httplib.cpp)
  target_include_directories(skymind_httplib PUBLIC ${LLAMA_CPP_DIR}/vendor/cpp-httplib)
  target_compile_features(skymind_httplib PRIVATE cxx_std_17)
  target_compile_definitions(skymind_httplib PUBLIC CPPHTTPLIB_TCP_NODELAY=1)
  if(WIN32)
    target_link_libraries(skymind_httplib PUBLIC ws2_32)
    target_compile_definitions(skymind_httplib PUBLIC _WIN32_WINNT=0x0A00)
  endif()
  find_package(Threads REQUIRED)
  target_link_libraries(skymind_httplib PUBLIC Threads::Threads)
  if(MSVC)
    target_compile_options(skymind_httplib PRIVATE /w)
  endif()

  add_library(skymind_llama STATIC src/llama_engine.cpp)
  target_link_libraries(skymind_llama PUBLIC skymind_core llama)

  add_executable(skymind_server src/server_main.cpp)
  target_link_libraries(skymind_server PRIVATE skymind_llama skymind_httplib)

  if(SKYMIND_LLAMA_PREBUILT_DIR)
    # DLL'leri exe'nin yanina kopyala: llama.dll, ggml*.dll, libomp*.dll (CUDA arka ucu 700+ MB, istege bagli)
    file(GLOB _backend_dlls ${SKYMIND_LLAMA_PREBUILT_DIR}/llama.dll ${SKYMIND_LLAMA_PREBUILT_DIR}/ggml*.dll
                              ${SKYMIND_LLAMA_PREBUILT_DIR}/libomp*.dll)
    if(NOT SKYMIND_WITH_CUDA)
      list(FILTER _backend_dlls EXCLUDE REGEX "ggml-cuda\\.dll$")
    else()
      file(GLOB _cuda_rt ${SKYMIND_LLAMA_PREBUILT_DIR}/cudart64*.dll
                         ${SKYMIND_LLAMA_PREBUILT_DIR}/cublas*.dll)
      list(APPEND _backend_dlls ${_cuda_rt})
    endif()
    add_custom_command(TARGET skymind_server POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_backend_dlls} $<TARGET_FILE_DIR:skymind_server>
      COMMENT "llama.cpp DLL'leri kopyalaniyor")
  endif()
endif()

# --- testler -----------------------------------------------------------------------------------
if(SKYMIND_BUILD_TESTS)
  enable_testing()
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip)
  FetchContent_MakeAvailable(googletest)

  add_executable(skymind_tests
    tests/test_validate.cpp tests/test_prompt_metrics.cpp tests/test_service.cpp)
  target_link_libraries(skymind_tests PRIVATE skymind_core GTest::gtest_main)
  target_compile_definitions(skymind_tests PRIVATE
    SKYMIND_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    SKYMIND_CONFIGS="${CMAKE_CURRENT_SOURCE_DIR}/configs")
  include(GoogleTest)
  gtest_discover_tests(skymind_tests)
endif()
