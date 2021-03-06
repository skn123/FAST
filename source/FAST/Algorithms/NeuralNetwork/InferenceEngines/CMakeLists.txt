include(GenerateExportHeader)

if(FAST_MODULE_TensorFlow)
    include(${PROJECT_SOURCE_DIR}/cmake/ModuleTensorFlow.cmake)

    add_library(InferenceEngineTensorFlow SHARED TensorFlowEngine.hpp TensorFlowEngine.cpp TensorFlowEngine.hpp TensorFlowEngine.cpp)
    target_link_libraries(InferenceEngineTensorFlow FAST ${TensorFlow_LIBRARIES})
    if(FAST_BUILD_TensorFlow_CUDA)
        target_compile_definitions(InferenceEngineTensorFlow PRIVATE -D FAST_TENSORFLOW_CUDA)
    elseif(FAST_BUILD_TensorFlow_ROCm)
        target_compile_definitions(InferenceEngineTensorFlow PRIVATE -D FAST_TENSORFLOW_ROCM)
    endif()
    message("-- Linking to ${TensorFlow_LIBRARIES}")
    target_include_directories(InferenceEngineTensorFlow PRIVATE ${FAST_INCLUDE_DIRS} ${PROJECT_BINARY_DIR})
    generate_export_header(InferenceEngineTensorFlow EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/TensorFlowExport.hpp)
    add_dependencies(InferenceEngineTensorFlow tensorflow)
    fast_add_inference_engine(TensorFlow)
endif()
if(FAST_MODULE_TensorRT)
    include(${PROJECT_SOURCE_DIR}/cmake/ModuleTensorRT.cmake)

    add_library(InferenceEngineTensorRT SHARED TensorRTEngine.hpp TensorRTEngine.cpp)
    target_include_directories(InferenceEngineTensorRT PRIVATE ${FAST_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(InferenceEngineTensorRT FAST ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})
    generate_export_header(InferenceEngineTensorRT EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/TensorRTExport.hpp)

    fast_add_inference_engine(TensorRT)
endif()
if(FAST_MODULE_OpenVINO)
    include(${PROJECT_SOURCE_DIR}/cmake/ModuleOpenVINO.cmake)

    add_library(InferenceEngineOpenVINO SHARED OpenVINOEngine.hpp OpenVINOEngine.cpp)
    generate_export_header(InferenceEngineOpenVINO EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/OpenVINOExport.hpp)
    add_dependencies(InferenceEngineOpenVINO OpenVINO)
    target_include_directories(InferenceEngineOpenVINO PRIVATE ${FAST_INCLUDE_DIRS} ${FAST_INCLUDE_DIRS}/openvino/)
    if(WIN32)
        target_link_libraries(InferenceEngineOpenVINO FAST inference_engine.lib inference_engine_legacy.lib inference_engine_transformations.lib inference_engine_lp_transformations.lib ngraph.lib)
    else()
       target_link_libraries(InferenceEngineOpenVINO FAST libinference_engine.so libinference_engine_legacy.so libinference_engine_transformations.so libinference_engine_lp_transformations.so)
    endif()

    fast_add_inference_engine(OpenVINO)
endif()
