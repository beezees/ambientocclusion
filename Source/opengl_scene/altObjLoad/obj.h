//obj.h
//file from some dude on internets
#ifndef OBJ_H
#define OBJ_H

struct ObjVertex
{
   float x, y, z;
};

typedef ObjVertex ObjNormal;

struct ObjTexCoord
{
   float u, v;
};

struct ObjTriangle
{
   int Vertex[3];
   int Normal[3];
   int TexCoord[3];
};

struct ObjModel
{
   int nVertex, nNormal, nTexCoord, nTriangle;

   ObjVertex* VertexArray;
   ObjNormal* NormalArray;
   ObjTexCoord* TexCoordArray;
   ObjTriangle* TriangleArray;
};

ObjModel* ObjLoadModel(char*, size_t);
size_t    ObjLoadFile(char*, char**);

#endif
