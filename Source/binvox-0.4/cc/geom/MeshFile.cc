//
// binvox, a binary 3D mesh voxelizer
// Copyright (c) 2004-2007 by Patrick Min, patrick.n.min "at" gmail "dot" com
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
// 
//
// $Id: MeshFile.cc,v 1.7 2007/01/19 13:22:38 min Exp min $
//

#include <algorithm>
#include <assert.h>
#include <math.h>
#if LINUX
#include <sys/resource.h>
#endif
#include "cc/file/Tokenfile.h"
#include "cc/time/WallTimer.h"
#include "MeshFile.h"
#include "Vertex.h"
#include "Face.h"
#include "geom_defs.h"





MeshFile::MeshFile(Mesh& mesh_ref, string filename) :
  MeshRef(mesh_ref),
  my_filename(filename)
{
  
}  // constructor



MeshFile::~MeshFile()
{

}  // destructor


