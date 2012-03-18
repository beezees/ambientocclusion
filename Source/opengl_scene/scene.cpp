//scene.cpp
//Calina Copos & Afton Geil
//EEC277 Final Project
#include <GL/glut.h>
#include <math.h>
#include "include/glm/glm.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <png.h>

using namespace std;

//Window dimensions:
int winWidth = 800;
int winHeight = 600;

//Display list for the model:
int displayList;

//Object model:
GLMmodel* model;

//Number of divisions in each direction for voxel grid:
int grid = 256;

float zcount;

float objxMin, objxMax, objyMin, objyMax, objzMin, objzMax;

void init(void)
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);	//White background
	
	glEnable(GL_DEPTH_TEST);	//Activate depth visibiltiy routines

	glEnable(GL_BLEND);		//Colour blending enabled
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);			//The sun

	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT);
}



//This creates a PNG of the input values to visually test the accuracy of the data.
bool writeMonochromeBuffer(char *filename, double *image, int width, int height, double minPlane, double maxPlane)
{
 double m, M;
 m = 1e20;
 M = -1e20;

 for (int i = 0; i < width * height; i++)
 if (image[i] >= minPlane && image[i] <= maxPlane)
 {
   m = std::min(m, image[i]);
   M = std::max(M, image[i]);
 }

 double scale = (M - m) / 255.0;

 unsigned char *buffer = new unsigned char[width * height * 3];

 unsigned char val;
 for (int i = 0; i < width * height; i++)
 if (image[i] < minPlane)
 {
   buffer[3 * i + 0] = 0;
   buffer[3 * i + 1] = 64;
   buffer[3 * i + 2] = 0;
 }
 else if (image[i] > maxPlane)
 {
   buffer[3 * i + 0] = 64;
   buffer[3 * i + 1] = 0;
   buffer[3 * i + 2] = 0;
 }
else
{
   val = (image[i] - m) / scale;
   buffer[3 * i + 0] = 0;
   buffer[3 * i + 1] = 0;
   buffer[3 * i + 2] = val;
 }

 unsigned char **rows  = new unsigned char*[height];
 for (int i = 0; i < height; i++)
   rows[i] = &buffer[width * i * 3];

 /* create file */
 FILE *fp = fopen(filename, "wb");
 if (!fp)
 {
   printf("File %s could not be opened for writing. Aborting.\n", filename);
   delete [] rows;
   delete [] buffer;
   return false;
 }

 /* initialize stuff */
 png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

 if (!png_ptr)
 {
   printf("png_create_write_struct failed\n.");
   delete [] rows;
   delete [] buffer;
   return false;
 }

 png_infop info_ptr = png_create_info_struct(png_ptr);
 if (!info_ptr)
 {
   printf("png_create_info_struct failed\n");
   delete [] rows;
   delete [] buffer; // Stay away from mah brackets!
   return false;
 }

 if (setjmp(png_jmpbuf(png_ptr)))
 {
   printf("LibPNG longjump abort! RUN FOR YOUR LIVES!\n");
   delete [] rows;
   delete [] buffer;
   return false;
 }

 png_init_io(png_ptr, fp);

 png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
   PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

 png_write_info(png_ptr, info_ptr);

 png_write_image(png_ptr, rows);

 png_write_end(png_ptr, NULL);

 delete [] rows;
 delete [] buffer;
 fclose(fp);

 cout << "Max:" << M << "min:" << m << endl;

 return true;
}



void convertToObj(int x, int y, double obj[])
{
        int viewport[4];
        double modelview[16];
        double projection[16];
        float z;
        double objx, objy, objz;

        glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
        glGetDoublev( GL_PROJECTION_MATRIX, projection );
        glGetIntegerv( GL_VIEWPORT, viewport );

        glReadPixels( x, viewport[3]-y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z);
        gluUnProject( x, viewport[3]-y, z, modelview, projection, viewport, &objx, &objy, &objz);

        obj[0] = objx;
        obj[1] = objy;
        obj[2] = objz;

	return;
}

void convertToVoxel(double objx, double objy, double objz, int vox[])
{
	double voxSize;
	double voxSizex, voxSizey, voxSizez;

	//Find dimensions of a single voxel:
	voxSize = 2.0/grid;
	voxSizex = (objxMax - objxMin)/grid;
	voxSizey = (objyMax - objyMin)/grid;
	voxSizez = (objzMax - objzMin)/grid;

	//Coordinates of containing voxel:
/*	vox[0] = floor((objx + 1.0f)/voxSize);
	if (vox[0] == 256) vox[0] = 255;
	vox[1] = floor((objy + 1.0f)/voxSize);
        if (vox[1] == 256) vox[1] = 255;
	vox[2] = floor((objz + 1.0f)/voxSize);
        if (vox[2] == 256) vox[2] = 255;
*/	vox[0] = floor((objx - objxMin)/voxSizex);
	if (vox[0] == 256) vox[0] = 255;
	vox[1] = floor((objy - objyMin)/voxSizey);
        if (vox[1] == 256) vox[1] = 255;
	vox[2] = floor((objz - objzMin)/voxSizez);
        if (vox[2] == 256) vox[2] = 255;

	return;
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(50.0, 1.0, 1.0, 1000.0);        //Set the field-of-view angle, aspect ratio, near and far clipping planes relative to camera position defined in gluLookAt.

	glMatrixMode(GL_MODELVIEW);	//Matrix describing transformations
	glLoadIdentity();
	gluLookAt(0, 0, 4, 0, 0, 0, 0, 1, 0);	//point to look from, at, upward direction

	//Lighting Stuff:
	float light0Direction[] = {10.0, 200.0, 100.0, 0.0};	//Distant light
	float whiteColour[] = {1.0, 1.0, 1.0, 1.0};
	float grayColour[] = {0.5, 0.5, 0.5, 0.5};
	float blackColour[] = {0.0, 0.0, 0.0, 1.0};
	glLightfv(GL_LIGHT0, GL_POSITION, light0Direction);
	glLightfv(GL_LIGHT0, GL_AMBIENT, blackColour);
	glLightfv(GL_LIGHT0, GL_SPECULAR, blackColour);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, grayColour);

	glCallList(displayList);

	glFlush();

	//Convert from window location to closest voxel vertex location:
        int pixelX, pixelY;
	double obj[3];
        double objx, objy, objz;
        int vox[3];
        int voxX, voxY, voxZ;
	double depth[winWidth*winHeight];
	ofstream filledVox;
        filledVox.open("filledVox.dat", ios::out);
        for (pixelY = 1; pixelY < winHeight; pixelY++){
		for (pixelX = 0; pixelX < winWidth; pixelX++){
        		convertToObj(pixelX, pixelY, obj);
			zcount++;
        		objx = obj[0];
		        objy = obj[1];
		        objz = obj[2];
			depth[pixelX+pixelY*winWidth] = objz;
//			cout << objz << endl;
			if (objz >= -1.0f && objz <= 1.0f){
				//TODO: find closest vertex and return normal to write to file
//				filledVox << objx << " " << objy << " " << objz << endl;
		        	convertToVoxel(objx, objy, objz, vox);
		        	voxX = vox[0];
		        	voxY = vox[1];
		        	voxZ = vox[2];
				filledVox << voxX << " " << voxY << " " << voxZ << endl;
			}
		}
	}
	filledVox.close();
	cout << zcount;

	writeMonochromeBuffer("zpic.png", depth, winWidth, winHeight, -1, 1);
	cout << "done";
	glutSwapBuffers();
	getchar();	//breakpoint

}

GLfloat arrayMax(GLfloat array[], GLuint numElements)
{
	GLfloat max = 0;
	GLuint i;
	for (i = 0; i < numElements; i++){
		if (array[i] > max){
			max = array[i];
		}
	}
	return max;
}

GLfloat arrayMin(GLfloat array[], GLuint numElements)
{
        GLfloat min = sizeof(GLfloat);
        GLuint i;
        for (i = 0; i < numElements; i++){
                if (array[i] < min){
                        min = array[i];
                }
        }
        return min;
}

int main(int argc, char** argv)
{
	//Initialize:
	glutInit(&argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(winWidth, winHeight);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutCreateWindow("So fluffyyyyyy!!!");
	init();

	//Import the .obj file:
	model = (GLMmodel*)malloc(sizeof(GLMmodel));
	model = glmReadOBJ("./objs/bunny.obj");

	//Obtain useful information from model:
	GLuint numvertices;
	numvertices = model->numvertices;
	printf("Number vertices: = %d \n", numvertices);
	GLfloat* vertices;
	vertices = (GLfloat*)malloc(sizeof(GLfloat) * 3 * (numvertices + 1));
	vertices = model->vertices;

	//Separate vertex x,y,z values:
	GLfloat modelx[numvertices];
	GLfloat modely[numvertices];
	GLfloat modelz[numvertices];
	GLuint i;
	for (i = 0; i < numvertices; i++){
		modelx[i] = *(vertices + 3*i);
		modely[i] = *(vertices + 3*i + 1);
		modelz[i] = *(vertices + 3*i + 2);
	}

	//Find model normals:
	GLfloat angle = 90.0;
	glmFacetNormals(model);
	glmVertexNormals(model, angle);

	//Find maximum and minimum coordinates:
//	GLfloat xObjMax, yObjMax, zObjMax;
//	GLfloat xObjMin, yObjMin, zObjMin;
	objxMax = arrayMax(modelx, numvertices);
	objyMax = arrayMax(modely, numvertices);
	objzMax = arrayMax(modelz, numvertices);
	objxMin = arrayMin(modelx, numvertices);
	objyMin = arrayMin(modely, numvertices);
	objzMin = arrayMin(modelz, numvertices);

	//Output the ranges of coordinate values:
/*	cout << "Min x:" << xObjMin << "\n";
	cout << "Min y:" << yObjMin << "\n";
	cout << "Min z:" << zObjMin << "\n";
	cout << "Max x:" << xObjMax << "\n";
	cout << "Max y:" << yObjMax << "\n";
	cout << "Max z:" << zObjMax << "\n";
*/
	//Load model into display list:
	displayList = glGenLists(1);
	glNewList(displayList, GL_COMPILE);
		glColor4f(1.0, 0.0, 1.0, 1.0);
		glmList(model, GLM_SMOOTH);
	glEndList();

/*	//Open file containing filled voxels and write to array:
	string str;
	string coord;
	int item;
	int (*voxels)[256][256];	//matrix of all possible voxel locations containing their status (empty = 0, filled =1)
	voxels = new int[256][256][256];
	ifstream vox_coords;
	vox_coords.open("vox_coords.dat", ios::in);
	int coordIndex = 0;
	int xin, yin, zin;
	while (!vox_coords.eof()){
		getline(vox_coords, str);
		istringstream coord_iss(str);
		while(coord_iss >> item){
			coordIndex++;
			if (coordIndex % 3 == 1) {
				xin = item;
//				cout << item << ' ';
			}
			if (coordIndex % 3 == 2) {
				yin = item;
//				cout << item << ' ';
			}
			if (coordIndex % 3 == 0){
				zin = item;
				voxels[xin][yin][zin] = 1;
//				cout << item << "\n";
			}
		}
	}
	vox_coords.close();
	int test = voxels[0][145][130];
	cout << test << '\n';
	test = voxels[6][179][145];
	cout << test << '\n';
*/
	//Callback functions:
	glutDisplayFunc(display);
	glutMainLoop();

	glmDelete(model);
//	delete [] voxels;

	return 0;
}

