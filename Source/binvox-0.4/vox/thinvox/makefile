#
# $Id$
#

TARGET =	thinvox

SRCHOME =	../..

CCSRCS =	$(TARGET).cc \
		$(SRCHOME)/cc/math/common.cc \
		$(SRCHOME)/cc/math/Vector.cc \
		$(SRCHOME)/cc/math/Matrix.cc \
		$(SRCHOME)/cc/math/MyRandom.cc \
		$(SRCHOME)/cc/time/WallTimer.cc \
		$(SRCHOME)/cc/time/MyTime.cc \
		$(SRCHOME)/cc/voxel/VoxelFile.cc \
		$(SRCHOME)/cc/voxel/VoxelFilter.cc \
		$(SRCHOME)/cc/voxel/PalagyiFilter.cc \
		$(SRCHOME)/cc/voxel/Template.cc \
		$(SRCHOME)/cc/voxel/Templates.cc \
		$(SRCHOME)/cc/voxel/VoxelRef.cc \
		$(SRCHOME)/cc/voxel/Voxels.cc

USER_FLAGS =	-Wno-write-strings

include $(SRCHOME)/cc/makefile.std



#
# main target
#

$(TARGET):	version.h $(OBJECTS)
	$(CXX) $(OUTPUT) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) $(LIBS)



ifeq ($(OS), WINDOWS)
clean:
	rm -rf *.obj *~
else
clean:
	rm -rf *.o *~ core
endif


depend:
	$(CC) -I$(INCLUDEPATH) -MM $(CCSRCS) 1> .depend


# DO NOT DELETE

include .depend

