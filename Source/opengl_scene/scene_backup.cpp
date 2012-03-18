//scene_backup.cpp
//Calina Copos and Afton Geil
//EEC 277 Final Project
//Loads bunny model and outputs filled voxels that project to screen.
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

int zcount;

void init(void)
{
glClearColor(1.0f, 1.0f, 1.0f, 1.0f); //White background

glEnable(GL_DEPTH_TEST); //Activate depth visibiltiy routines
// glDepthRange(0,1); //Maps window coordinate z to range [0,1]

glEnable(GL_BLEND); //Colour blending enabled
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

glEnable(GL_LIGHTING);
glEnable(GL_LIGHT0); //The sun

glEnable(GL_COLOR_MATERIAL);
glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT);
}




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

 unsigned char **rows = new unsigned char*[height];
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
        zcount++;
        gluUnProject( x, viewport[3]-y, z, modelview, projection, viewport, &objx, &objy, &objz);
// printf("In world coords (x,y,z) = %f %f %f \n", objx, objy, objz);
// cout << objx << " " << objy << " " << objz << "\n";

        obj[0] = objx;
        obj[1] = objy;
        obj[2] = objz;

return;
}


void display(void)
{
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

float zPlane = 1;
float lower, upper;
float near, far;
lower = (zPlane == 0) ? zPlane - 10:(zPlane - 2*(zPlane < 0 ? -zPlane : zPlane));
upper = (zPlane == 0) ? zPlane + 10:(zPlane + 2*(zPlane < 0 ? -zPlane : zPlane));
near = upper - zPlane - .00001;
far = upper - zPlane + .00001;

glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(50.0, 1.0, 1.0, 1000.0); //Set the field-of-view angle, aspect ratio, near and far clipping planes relative to camera position defined in gluLookAt.

glMatrixMode(GL_MODELVIEW); //Matrix describing transformations
glLoadIdentity();
gluLookAt(2, 2, 2, 0, 0, 0, 0, 1, 0); //point to look from, at, upward direction

//Lighting Stuff:
float light0Direction[] = {10.0, 200.0, 100.0, 0.0}; //Distant light
float whiteColour[] = {1.0, 1.0, 1.0, 1.0};
float grayColour[] = {0.5, 0.5, 0.5, 0.5};
float blackColour[] = {0.0, 0.0, 0.0, 1.0};
glLightfv(GL_LIGHT0, GL_POSITION, light0Direction);
glLightfv(GL_LIGHT0, GL_AMBIENT, blackColour);
glLightfv(GL_LIGHT0, GL_SPECULAR, blackColour);
glLightfv(GL_LIGHT0, GL_DIFFUSE, grayColour);

glCallList(displayList);

/* glMatrixMode(GL_PROJECTION);
glLoadIdentity();
glOrtho(-800/2,800/2,-800/2,800/2,1,20);
glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
glColor4f(.5,.6,1,1);
glTranslated(0,0,-1);
glBegin(GL_QUADS);
glVertex3f(0,0,-1);
glVertex3f(800/2,0,-7);
glVertex3f(800/2,800/2,-7);
glVertex3f(0,800/2,-1);
glEnd();
*/
glFlush();

//Convert from window location to closest voxel vertex location:
        int pixelX, pixelY;
double obj[3];
        double objx, objy, objz;
        int vox[3];
        int voxX, voxY, voxZ;
float numVox = 0;
double depth[winWidth*winHeight];
ofstream filledVox;
        filledVox.open("filledVox.dat", ios::out);
        for (pixelY = 0; pixelY < winHeight; pixelY++){ //the 2 and 3 are a ghetto fix to make the voxel y's on a scale from 0 to 255.
for (pixelX = 0; pixelX < winWidth; pixelX++){
         convertToObj(pixelX, pixelY, obj);
         objx = obj[0];
objy = obj[1];
objz = obj[2];
depth[pixelX+pixelY*winWidth] = objz;
/* convertToVoxel(objx, objy, objz, xObjMin, xObjMax, yObjMin, yObjMax, zObjMin, zObjMax, vox);
voxX = vox[0];
voxY = vox[1];
voxZ = vox[2];
int test = voxels[voxX][voxY][voxZ];
filledVox << test << "\n";
cout << voxX << " " << voxY << " " << voxZ << "\n";
if (voxels[voxX][voxY][voxZ] == 1){
//write voxel coordinates to file
filledVox << voxX << " " << voxY << " " << voxZ << endl;
numVox++;
}
*/ }
}

//Determine the correct grid dimensions required and ensure voxel data fills 3D matrix:
/* int gridDim;
gridDim = ceil(pow(numVox, (1/3)));
int extras;
extras = pow(gridDim, 3) - numVox;
while (extras >= 0){
filledVox<< -42 << " " << -42 << " " << -42 << endl;
extras--;
}
*/ filledVox.close();
cout << zcount;

writeMonochromeBuffer("zpic.png", depth, winWidth, winHeight, -1, 1);
       glutSwapBuffers();
getchar();

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

int IntArrayMin(int array[], int numElements)
{
        int min = sizeof(int);
        int i;
        for (i = 0; i < numElements; i++){
                if (array[i] < min){
                        min = array[i];
                }
        }
        return min;
}

void convertToVoxel(double objx, double objy, double objz, GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax, int vox[])
{
double voxSizeX, voxSizeY, voxSizeZ;

//Find dimensions of a single voxel:
voxSizeX = (xmax - xmin)/grid;
voxSizeY = (ymax - ymin)/grid;
voxSizeZ = (zmax - zmin)/grid;

//Coordinates of containing voxel:
vox[0] = floor((objx-xmin)/voxSizeX);
vox[1] = floor((objy-ymin)/voxSizeY);
vox[2] = floor((objz-zmin)/voxSizeZ);

return;
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
GLfloat x[numvertices];
GLfloat y[numvertices];
GLfloat z[numvertices];
GLuint i;
for (i = 0; i < numvertices; i++){
x[i] = *(vertices + 3*i);
y[i] = *(vertices + 3*i + 1);
z[i] = *(vertices + 3*i + 2);
}

//Find maximum and minimum coordinates:
GLfloat xObjMax, yObjMax, zObjMax;
GLfloat xObjMin, yObjMin, zObjMin;
xObjMax = arrayMax(x, numvertices);
yObjMax = arrayMax(y, numvertices);
zObjMax = arrayMax(z, numvertices);
xObjMin = arrayMin(x, numvertices);
yObjMin = arrayMin(y, numvertices);
zObjMin = arrayMin(z, numvertices);

//Output the ranges of coordinate values:
cout << "Min x:";
cout << xObjMin << "\n";
cout << "Min y:";
cout << yObjMin << "\n";
cout << "Min z:";
cout << zObjMin << "\n";
cout << "Max x:";
cout << xObjMax << "\n";
cout << "Max y:";
cout << yObjMax << "\n";
cout << "Max z:";
cout << zObjMax << "\n";

//Display model:
displayList = glGenLists(1);
glNewList(displayList, GL_COMPILE);
glColor4f(1.0, 0.0, 1.0, 1.0);
glmList(model, GLM_NONE);
glEndList();
/*
//Open file containing filled voxels and write to array:
string str;
string coord;
int item;
int (*voxels)[256][256]; //matrix of all possible voxel locations containing their status (empty = 0, filled =1)
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
// cout << item << ' ';
}
if (coordIndex % 3 == 2) {
yin = item;
// cout << item << ' ';
}
if (coordIndex % 3 == 0){
zin = item;
voxels[xin][yin][zin] = 1;
// cout << item << "\n";
}
}
}
vox_coords.close();
// int test = voxels[0][145][130];
// cout << test << '\n';
// test = voxels[6][179][145];
// cout << test << '\n';

//Convert from window location to closest voxel vertex location:
int pixelX, pixelY;
double obj[3];
double objx, objy, objz;
int vox[3];
int voxX, voxY, voxZ;
float numVox = 0;
ofstream filledVox;
filledVox.open("filledVox.dat", ios::out);
for (pixelY = 0; pixelY < winHeight; pixelY++){ //the 2 and 3 are a ghetto fix to make the voxel y's on a scale from 0 to 255.
for (pixelX = 0; pixelX < winWidth; pixelX++){
convertToObj(pixelX, pixelY, obj);
objx = obj[0];
objy = obj[1];
objz = obj[2];
convertToVoxel(objx, objy, objz, xObjMin, xObjMax, yObjMin, yObjMax, zObjMin, zObjMax, vox);
voxX = vox[0];
voxY = vox[1];
voxZ = vox[2];
int test = voxels[voxX][voxY][voxZ];
filledVox << test << "\n";
cout << voxX << " " << voxY << " " << voxZ << "\n";
if (voxels[voxX][voxY][voxZ] == 1){
//write voxel coordinates to file
filledVox << voxX << " " << voxY << " " << voxZ << endl;
numVox++;
}
}
}

//Determine the correct grid dimensions required and ensure voxel data fills 3D matrix:
int gridDim;
gridDim = ceil(pow(numVox, (1/3)));
int extras;
extras = pow(gridDim, 3) - numVox;
while (extras >= 0){
filledVox<< -42 << " " << -42 << " " << -42 << endl;
extras--;
}
filledVox.close();
cout << zcount;
*/
//Callback functions:
glutDisplayFunc(display);
glutMainLoop();

glmDelete(model);


return 0;
}
