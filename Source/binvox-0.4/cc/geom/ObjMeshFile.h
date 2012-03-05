//
// $Id$
//

#ifndef __ObjMeshFile_h
#define __ObjMeshFile_h

#include "MeshFile.h"





class ObjMeshFile : public MeshFile
{

public:

  ObjMeshFile(Mesh& mesh_ref, string filename);
  ~ObjMeshFile();

  int load();
  
  
private:
  

};  // ObjMeshFile class


#endif


