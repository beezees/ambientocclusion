//main.cpp
//file from some dude on internets
#include <stdlib.h>
#include <stdio.h>
#include "obj.h"

int main()
{
   char* memory = NULL;
   size_t bytes = ObjLoadFile("objs/bunny.obj", &memory);

   ObjModel* model = ObjLoadModel(memory, bytes);

   printf("Object Model has: %d faces!\n", model->nTriangle);
   
   free(model->NormalArray);
   free(model->TexCoordArray);
   free(model->TriangleArray);
   free(model->VertexArray);
   free(model);

   return 0;
}

