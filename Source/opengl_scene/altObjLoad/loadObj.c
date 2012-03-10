//Code from some dude
typedef struct _ObjVertex {
     float X, Y, Z;
} ObjVertex;
typedef _ObjVertex _ObjNormal;

typedef struct _ObjTexCoord {
     float U, V;
} ObjTexCoord;

typedef struct _ObjTriangle {
     int Vertex[3];
     int Normal[3];
     int TexCoord[3];
} ObjTriangle;


typedef struct _ObjModel {
     int NumVertex, NumNormal, NumTexCoord, NumTriangle;

     ObjVertex *VertexArray;
     ObjNormal *NormalArray;
     ObjTexCoord *TexCoordArray;

     ObjTriangle *TriangleArray;
} ObjModel;
/* Return 1 if strings are equal, 0 if not */
/* Make sure we don't ever pass the given end (E) */
int StrEqual(char *A, char *B, char *E, int count)
{
	int c;
	c = 0;
	while ((c < count) && (A != E))
	{
		 if (A[c] != B[c]) return 0;
		 c++;
     }
     if (A == E) return 0;
      else return 1;
}

/* Return 1 if the character is dead space, 0 if not */
int IsNumeric(char A)
{
	if (A == '.') return 1;
	if (A == '-') return 1;
	if ((A >= 0x30) && (A <= 0x39)) return 1;
	return 0;
}


/* Return 1 if the character is dead space, 0 if not */
int IsDeadSpace(char A)
{
    	if (A < 33) return 1;
	 else return 0;
}

/* Return 1 if the character is a newline/linefeed, 0 if not */
int IsEOL(char A)
{
	if (A == 10) return 1;
	 else return 0;
}
ObjModel* ObjLoadModel(char *mem, int sz)
{
	char *p, *e;
	char b[512];
	int c;
	ObjModel *ret;
	
	// the returned model struct, allocate and clear
	ret = calloc(1,sizeof(ObjModel));
	
	// current position and end location pointers
	p = mem;
	e = mem + sz;
	
	// first pass, scan the number of vertex, normals, texcoords, and faces
	while (p != e)
	{
		 // nibble off one line, ignoring leading dead space
		 c = 0;
		 while ((IsDeadSpace(*p)) && (p != e)) p++;
		 while ((!IsEOL(*p)) && (p != e) && (c < 512)) { b[c++] = *p; p++; }
		 
		 // ok, b[] contains the current line
		 if (StrEqual(b,"vn",&b[c],2)) ret->NumNormal++;
		  else
		 if (StrEqual(b,"vt",&b[c],2)) ret->NumTexCoord++;
		  else
		 if (StrEqual(b,"v",&b[c],1)) ret->NumVertex++;
		  else
		 if (StrEqual(b,"f",&b[c],1)) ret->NumTriangle++;
     }
     
     // now allocate the arrays
     ret->VertexArray = malloc(sizeof(ObjVertex)*ret->NumVertex);
     ret->NormalArray = malloc(sizeof(ObjNormal)*ret->NumNormal);
     ret->TexCoordArray = malloc(sizeof(ObjTexCoord)*ret->NumTexCoord);
     ret->TriangleArray = malloc(sizeof(ObjTriangle)*ret->NumTriangle);
	// finally, go back and scan the values
	p = mem;
	Vc = Nc = Tc = Fc = 0;
	
	while (p != e)
	{
		 // nibble off one line, ignoring leading dead space
		 c = 0;
		 while ((IsDeadSpace(*p)) && (p != e)) p++;
		 while ((!IsEOL(*p)) && (p != e) && (c < 512)) { b[c++] = *p; p++; }

		 // ok, b[] contains the current line
		 if (StrEqual(b,"vn",&b[c],2))
		 {
			sscanf(b,"vn %f %f %f",&ret->NormalArray[Nc].X,&ret->NormalArray[Nc].Y,&ret->NormalArray[Nc].Z);
			Nc++;
           }
		  else
		 if (StrEqual(b,"vt",&b[c],2))
		 {
			sscanf(b,"vt %f %f",&ret->TexCoordArray[Tc].U,&ret->TexCoordArray[Tc].V);
			Tc++;
           }
		  else
		 if (StrEqual(b,"v",&b[c],1))
		 {
			sscanf(b,"v %f %f %f",&ret->VertexArray[Vc].X,&ret->VertexArray[Vc].Y,&ret->VertexArray[Vc].Z);
			Vc++;
           }
		  else
		 if (StrEqual(b,"f",&b[c],1))
		 {
			sscanf(b,"f %d/%d/%d %d/%d/%d %d/%d/%d",
                      &ret->TriangleArray[Fc].Vertex[0],&ret->TriangleArray[Fc].TexCoord[0],&ret->TriangleArray[Fc].Normal[0],
				  &ret->TriangleArray[Fc].Vertex[1],&ret->TriangleArray[Fc].TexCoord[1],&ret->TriangleArray[Fc].Normal[1],
				  &ret->TriangleArray[Fc].Vertex[2],&ret->TriangleArray[Fc].TexCoord[2],&ret->TriangleArray[Fc].Normal[2]);
			Fc++;
           }
     }
     
     return ret;
}
