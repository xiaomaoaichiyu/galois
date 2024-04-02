add_executable(gnn gnn.cpp)
add_dependencies(apps gnn)
target_link_libraries(gnn PRIVATE Galois::shmem lonestar)
install(TARGETS gnn DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

# add_test_scale(small gnn -transposedGraph -tolerance=0.01 "${BASEINPUT}/scalefree/transpose/rmat10.tgr")
# add_test_scale(small-topo gnn -transposedGraph -tolerance=0.01 -algo=Topo "${BASEINPUT}/scalefree/transpose/rmat10.tgr")
