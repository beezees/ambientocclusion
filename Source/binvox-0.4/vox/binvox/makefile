#
# $Id$
#

TARGET =	binvox

SRCHOME =	../..

CCSRCS =	$(TARGET).cc \
		$(SRCHOME)/cc/file/Buffer.cc \
		$(SRCHOME)/cc/file/Datafile.cc \
		$(SRCHOME)/cc/file/Tokenfile.cc \
		$(SRCHOME)/cc/geom/Face.cc \
		$(SRCHOME)/cc/geom/Mesh.cc \
		$(SRCHOME)/cc/geom/MeshFile.cc \
		$(SRCHOME)/cc/geom/MeshFileIdentifier.cc \
		$(SRCHOME)/cc/geom/MeshRef.cc \
		$(SRCHOME)/cc/geom/MeshView.cc \
		$(SRCHOME)/cc/geom/ObjMeshFile.cc \
		$(SRCHOME)/cc/geom/PlyMeshFile.cc \
		$(SRCHOME)/cc/geom/Vertex.cc \
		$(SRCHOME)/cc/math/common.cc \
		$(SRCHOME)/cc/math/Matrix.cc \
		$(SRCHOME)/cc/math/MyRandom.cc \
		$(SRCHOME)/cc/math/Vector.cc \
		$(SRCHOME)/cc/time/WallTimer.cc \
		$(SRCHOME)/cc/time/MyTime.cc \
		$(SRCHOME)/cc/ui/Camera.cc \
		$(SRCHOME)/cc/ui/CameraList.cc \
		$(SRCHOME)/cc/ui/GLwindow.cc \
		$(SRCHOME)/cc/ui/opengl.cc \
		$(SRCHOME)/cc/ui/View.cc \
		$(SRCHOME)/cc/voxel/VoxelFile.cc \
		$(SRCHOME)/cc/voxel/Voxelizer.cc \
		$(SRCHOME)/cc/voxel/VoxelFilter.cc \
		$(SRCHOME)/cc/voxel/VoxelRef.cc \
		$(SRCHOME)/cc/voxel/Voxels.cc

CSRCS =		$(SRCHOME)/cc/ply/ply_gatech.c

OPENGL =	1
WIN_SYSTEM =	1

USER_FLAGS =	-Wno-write-strings

include $(SRCHOME)/cc/makefile.std



#
# main target
#

$(TARGET):	version.h $(OBJECTS)
	$(CXX) $(OUTPUT) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) $(LIBS)

ifeq ($(OS), WINDOWS)
clean:
	rm -rf *~ $(OBJECTS)
else
clean:
	rm -rf *~ core $(OBJECTS)
endif


depend:
	$(CC) -I$(INCLUDEPATH) -MM $(CCSRCS) 1> .depend


# DO NOT DELETE

include .depend

