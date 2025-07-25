cmake_minimum_required(VERSION 2.8.3...3.28.3)

if(NOT FAST_DOWNWARD_MAIN_CMAKELISTS_READ)
    message(
        FATAL_ERROR
        "Run cmake on the CMakeLists.txt in the 'src' directory, "
        "not the one in 'src/search'. Please delete CMakeCache.txt "
        "from the current directory and restart cmake.")
endif()


## == Project ==

project(downward)
fast_downward_set_compiler_flags()
fast_downward_set_linker_flags()

# Collect source files needed for the active plugins.
include("${CMAKE_CURRENT_SOURCE_DIR}/DownwardFiles.cmake")
add_executable(downward ${PLANNER_SOURCES})

# Random walker executable
#set(WALKER_SOURCES ${PLANNER_SOURCES})
#list(REMOVE_ITEM WALKER_SOURCES "planner.cc")
#list(APPEND WALKER_SOURCES "random_walker.cc")
#add_executable(random_walker ${WALKER_SOURCES})
add_executable(random_walker axioms.cc causal_graph.cc int_packer.cc global_operator.cc global_state.cc globals.cc state_registry.cc task_proxy.cc task_tools.cc successor_generator.cc options/bounds.cc options/doc_printer.cc options/doc_store.cc options/errors.cc options/option_parser.cc options/plugin.cc options/registries.cc tasks/root_task.cc utils/rng.cc utils/system.cc utils/system_unix.cc utils/system_windows.cc utils/timer.cc random_walker.cc)

# Feature extractor executable
add_executable(feature_extractor axioms.cc causal_graph.cc domain_transition_graph.cc evaluation_context.cc evaluation_result.cc int_packer.cc global_operator.cc global_state.cc globals.cc heuristic.cc heuristic_cache.cc scalar_evaluator.cc state_registry.cc task_proxy.cc task_tools.cc state_encoder.cc successor_generator.cc heuristics/additive_heuristic.cc heuristics/cea_heuristic.cc heuristics/cea_heuristic_f.cc heuristics/ff_heuristic.cc heuristics/ff_heuristic_f.cc heuristics/relaxation_heuristic.cc options/bounds.cc options/doc_printer.cc options/doc_store.cc options/errors.cc options/option_parser.cc options/plugin.cc options/registries.cc tasks/root_task.cc utils/rng.cc utils/system.cc utils/system_unix.cc utils/system_windows.cc utils/timer.cc feature_extractor.cc)

## == Includes ==

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ext)

## == Libraries ==

# On Linux, find the rt library for clock_gettime().
if(UNIX AND NOT APPLE)
    target_link_libraries(downward rt)
endif()

# On Windows, find the psapi library for determining peak memory.
if(WIN32)
    target_link_libraries(downward psapi)
endif()

# If any enabled plugin requires an LP solver, compile with all
# available LP solvers. If no solvers are installed, the planner will
# still compile, but using heuristics that depend on an LP solver will
# cause an error. This behavior can be overwritten by setting the
# option USE_LP to false.
option(
  USE_LP
  "Compile with support for all LP solvers installed on this system."
  TRUE)

if(PLUGIN_LP_SOLVER_ENABLED AND USE_LP)
    find_package(OSI OPTIONAL_COMPONENTS Cpx Clp Grb)
    if(OSI_FOUND AND (OSI_Cpx_FOUND OR OSI_Clp_FOUND OR OSI_Grb_FOUND))
        foreach(SOLVER Cpx Clp Grb)
            if(OSI_${SOLVER}_FOUND)
                string(TOUPPER ${SOLVER} TMP_SOLVER_UPPER_CASE)
                mark_as_advanced(TMP_SOLVER_UPPER_CASE)
                add_definitions("-D COIN_HAS_${TMP_SOLVER_UPPER_CASE}")
                include_directories(${OSI_${SOLVER}_INCLUDE_DIRS})
                target_link_libraries(downward ${OSI_${SOLVER}_LIBRARIES})
            endif()
        endforeach()

        # Note that basic OSI libs must be added after (!) all OSI solver libs.
        add_definitions("-D USE_LP")
        include_directories(${OSI_INCLUDE_DIRS})
        target_link_libraries(downward ${OSI_LIBRARIES})
    endif()

    if(OSI_Cpx_FOUND AND CPLEX_RUNTIME_LIBRARY)
        add_custom_command(TARGET downward POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CPLEX_RUNTIME_LIBRARY}
            $<TARGET_FILE_DIR:downward>
        )
    endif()
endif()

if ($ENV{DOWNWARD_USE_ISNAN_WORKAROUND})
    set(USE_ISNAN_WORKAROUND_DEFAULT_VALUE TRUE)
else()
    set(USE_ISNAN_WORKAROUND_DEFAULT_VALUE FALSE)
endif()

option(
  USE_ISNAN_WORKAROUND
  "With some compilers this workaround fixes compilation errors involving the isnan stdlib function."
  ${USE_ISNAN_WORKAROUND_DEFAULT_VALUE})

message("USE_ISNAN_WORKAROUND: ${USE_ISNAN_WORKAROUND}")
if(USE_ISNAN_WORKAROUND)
    add_definitions("-D USE_ISNAN_WORKAROUND")
endif()

# Additions for the learned heuristic. Temporary.
#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../learned-heuristic")
#target_link_libraries(downward ${CMAKE_CURRENT_SOURCE_DIR}/../../../learned-heuristic/liblearned_evaluator.a)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../../simple-nn")
target_link_libraries(downward ${CMAKE_CURRENT_SOURCE_DIR}/../../../simple-nn/nn.a)
