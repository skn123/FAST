if(APPLE)
    file(GLOB installedSOs
            "$ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX}/fast/lib/*.dylib*"
            "$ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX}/fast/lib/*.so*")
    foreach(SO ${installedSOs})
        if(NOT IS_SYMLINK ${SO})
            message("-- Setting runtime path of ${SO}")
            execute_process(COMMAND codesign --remove-signature ${SO}) # adding rpath makes any signed binaries invalid which will make macos complain
            execute_process(COMMAND install_name_tool -add_rpath "@loader_path/../lib" ${SO})
        endif()
    endforeach()
else()
    file(GLOB installedSOs "$ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX}/fast/lib/*.so*")
    foreach(SO ${installedSOs})
        message("-- Setting runtime path of ${SO}")
        execute_process(COMMAND patchelf --set-rpath "$ORIGIN/../lib" ${SO})
    endforeach()
endif()
