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

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(50.0, 1.0, 1.0, 10000.0);        //Set the field-of-view angle, aspect ratio, near and far clipping planes relative to camera position defined in gluLookAt.

	glMatrixMode(GL_MODELVIEW);	//Matrix describing transformations
	glLoadIdentity();
	gluLookAt(5, 5, 5, 0, 0, 0, 0, 1, 0);	//point to look from, at, upward direction

	//Lighting Stuff:
/*	float light0Direction[] = {10.0, 200.0, 100.0, 0.0};	//Distant light
	float whiteColour[] = {1.0, 1.0, 1.0, 1.0};
	float grayColour[] = {0.5, 0.5, 0.5, 0.5};
	float blackColour[] = {0.0, 0.0, 0.0, 1.0};
	glLightfv(GL_LIGHT0, GL_POSITION, light0Direction);
	glLightfv(GL_LIGHT0, GL_AMBIENT, blackColour);
	glLightfv(GL_LIGHT0, GL_SPECULAR, blackColour);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, grayColour);
*/
	glCallList(displayList);

	glFlush();
	glutSwapBuffers();
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
    //printf("In world coords (x,y,z) = %f %f %f \n", objx, objy, objz);

    obj[0] = objx;
    obj[1] = objy;
    obj[2] = objz;

    return;
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
	glutCreateWindow("Importing .obj");
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

	//Open file containing filled voxels and write to array:
	string str;
	string coord;
	int item;
	int (*voxels)[256][256];	//array of all possible voxel locations containing their status (empty = 0, filled =1)
	voxels = new int[256][256][256];
//	int* voxels;
//	voxels = (int*)calloc(256*256*256, sizeof(int));
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
//	int test = voxels[0][145][130];
//	cout << test << '\n';
//	test = voxels[6][179][145];
//	cout << test << '\n';

	//Convert from window location to closest voxel vertex location:
        int pixelX, pixelY;
	double obj[3];
        double objx, objy, objz;
        int vox[3];
        int voxX, voxY, voxZ;
	float numVox = 0;
	ofstream filledVox;
        filledVox.open("filledVox.dat", ios::out);
        for (pixelY = 3; pixelY < (winHeight - 2); pixelY++){	//the 2 and 3 are a ghetto fix to make the voxel y's on a scale from 0 to 255.
		for (pixelX = 1; pixelX < (winWidth - 1); pixelX++){
        		convertToObj(pixelX, pixelY, obj);
        		objx = obj[0];
		        objy = obj[1];
		        objz = obj[2];
		        convertToVoxel(objx, objy, objz, xObjMin, xObjMax, yObjMin, yObjMax, zObjMin, zObjMax, vox);
		        voxX = vox[0];
		        voxY = vox[1];
		        voxZ = vox[2];
//			if (voxZ < 0) voxZ = 145;
//			if (voxZ > 256) voxZ = 176;
			int test = voxels[voxX][voxY][voxZ];
			filledVox << test << "\n";
//		        cout << voxX << " " << voxY << " " << voxZ << "\n";
/*			if (voxels[voxX][voxY][voxZ] == 1){
				//write voxel coordinates to file
				filledVox << voxX << " " << voxY << " " << voxZ << endl;
				numVox++;
			}
*/		}
	}

	//Determine the correct grid dimensions required and ensure voxel data fills 3D matrix:
/*	int gridDim;
	gridDim = ceil(pow(numVox, (1/3)));
	int extras;
	extras = pow(gridDim, 3) - numVox;
	while (extras >= 0){
		filledVox<< -42 << " " << -42 << " " << -42 << endl;
		extras--;
	}
*/	filledVox.close();

	//Callback functions:
	glutDisplayFunc(display);
	glutMainLoop();

	glmDelete(model);
//	free(voxels);
	delete [] voxels;

	return 0;
}

