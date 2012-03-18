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
//	double voxSize;
	double voxSizex, voxSizey, voxSizez;

	//Find dimensions of a single voxel:
//	voxSize = 2.0/grid;
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

	glCallList(displayList);

	glFlush();

	int vox[3];
	double obj[3];
        ifstream results;
        results.open("nonzeroc.dat", ios::in);
        int coordIndex = 0;
        int xin, yin, zin;
        while (!results.eof()){
                getline(results, str);
                istringstream results_iss(str);
                while(results_iss >> item){
                        coordIndex++;
                        if (coordIndex % 4 == 1) {
                                vox[0] = item;
//                              cout << item << ' ';
                        }
                        if (coordIndex % 4 == 2) {
                                vox[1] = item;
//                              cout << item << ' ';
                        }
                        if (coordIndex % 4 == 3){
                                vox[2] = item;
//                              cout << item << "\n";
                        }
			if (coordIndex % 4 == 0){
				ao = item;
				cout << item << "\n";
			}
			convertToObj(vox[], obj[]);
			objx = obj[0];
			objy = obj[1];
			objz = obj[2];		
                }
        }
        results.close();

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
	objxMax = arrayMax(modelx, numvertices);
	objyMax = arrayMax(modely, numvertices);
	objzMax = arrayMax(modelz, numvertices);
	objxMin = arrayMin(modelx, numvertices);
	objyMin = arrayMin(modely, numvertices);
	objzMin = arrayMin(modelz, numvertices);

	//Output the ranges of coordinate values:
	cout << "Min x:" << objxMin << "\n";
	cout << "Min y:" << objyMin << "\n";
	cout << "Min z:" << objzMin << "\n";
	cout << "Max x:" << objxMax << "\n";
	cout << "Max y:" << objyMax << "\n";
	cout << "Max z:" << objzMax << "\n";

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

