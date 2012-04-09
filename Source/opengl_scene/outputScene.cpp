//outputScene.cpp
//Calina Copos & Afton Geil
//EEC277 Final Project
#define GL_GLEXT_PROTOTYPES
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

float objxMin, objxMax, objyMin, objyMax, objzMin, objzMax;

GLfloat AOMin, AOMax;

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

void convertToWindow(GLfloat obj[], GLfloat win[])
{
        int viewport[4];
        double modelview[16];
        double projection[16];
        double objx, objy, objz;
	double winx, winy, winz;

        objx = obj[0];
        objy = obj[1];
        objz = obj[2];

        glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
        glGetDoublev( GL_PROJECTION_MATRIX, projection );
        glGetIntegerv( GL_VIEWPORT, viewport );

        gluProject( objx, objy, objz, modelview, projection, viewport, &winx, &winy, &winz);

	win[0] = winx;
	win[1] = winy;
	win[2] = winz;

	return;
}


void convertToObj(GLfloat vox[], GLfloat obj[])
{
//	double voxSize;
	GLfloat voxSizex, voxSizey, voxSizez;

	//Find dimensions of a single voxel:
//	voxSize = 2.0/grid;
	voxSizex = (objxMax - objxMin)/grid;
	voxSizey = (objyMax - objyMin)/grid;
	voxSizez = (objzMax - objzMin)/grid;

	//Coordinates in object space:
	obj[0] = vox[0] * voxSizex - 1;
	obj[1] = vox[1] * voxSizey - 1;
	obj[2] = vox[2] * voxSizez - 1;

	return;
}

int intArrayMax(int array[], int numElements)
{
        int max = 0;
        int m;
        for (m = 0; m < numElements; m++){
                if (array[m] > max){
                        max = array[m];
                }
        }
        return max;
}

int intArrayMin(int array[], int numElements)
{
        int min = sizeof(int);
        int n;
        for (n = 0; n < numElements; n++){
                if (array[n] < min){
                        min = array[n];
                }
        }
        return min;
}

void findMinMaxAO(void)
{

        string str;
        double item;
        int *ao;
	ao = new int [winWidth*winHeight];
        ifstream results;
        results.open("nonzero_filledVox.dat", ios::in);
        int coordIndex = 0;
	int i = 0;
        while (!results.eof()){
                getline(results, str);
                istringstream results_iss(str);
                while(results_iss >> item){
                        coordIndex++;
                        if (coordIndex % 4 == 0){
                                ao[i] = item;
				i++;
                        }
                }
	}

	AOMin = intArrayMin(ao, i);
	AOMax = intArrayMax(ao, i);

	delete [] ao;
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

//	glCallList(displayList);

	glFlush();
	glutSwapBuffers();

	findMinMaxAO();

	string str;
	GLfloat item;
	GLfloat vox[3];
	GLfloat obj[3];
	GLfloat objx, objy, objz;
	GLfloat win[3];
	GLfloat rawAO;
	GLfloat ao;
	GLfloat oldColour;
        ifstream results;
        results.open("nonzero_filledVox.dat", ios::in);
        int coordIndex = 0;
        while (!results.eof()){
                getline(results, str);
                istringstream results_iss(str);
                while(results_iss >> item){
                        coordIndex++;
                        if (coordIndex % 4 == 1) {
                                vox[0] = item;
//				cout << item << endl;
                        }
                        if (coordIndex % 4 == 2) {
                                vox[1] = item;
//				cout << item << endl;
                        }
                        if (coordIndex % 4 == 3){
                                vox[2] = item;
//				cout << item << endl;
                        }
			if (coordIndex % 4 == 0){
				rawAO = item;
//				cout << item << endl;
			}
		}
		ao = (rawAO - AOMin)/ (AOMax - AOMin);	//normalize AO values
/*		cout << vox[0] << " "; 
		cout << vox[1] << " "; 
		cout << vox[2] << " ";
		cout << ao << endl;
*/		convertToObj(vox, obj);
		objx = obj[0];
		objy = obj[1];
		objz = obj[2];
/*		cout << objx << " " << objy << " ";
		cout << objz << endl;
*///		GLfloat colour[] = {0, ao * 255.0f, ao * 255.0f};	//scale AO for 8-bit RGB colour
//		cout << vox[0] << " "; 
		convertToWindow(obj, win);
//		cout << win[0] << " " << win[1] << " " << win[2] << endl;
		glReadPixels(win[0], win[1], 1, 1, GL_RGB, GL_FLOAT, &oldColour);
//		cout << oldColour << "R " << *(&oldColour + 1) << "G " << *(&oldColour + 2) << "B ";
		GLfloat colour[3];
		colour[0] = oldColour - ao;
		colour[1] = *(&oldColour) - ao;
		colour[2] = *(&oldColour) - ao;
//		glRasterPos3fv(obj);	//set the object space coordinate to occlude
//		glDrawPixels(1, 1, GL_RGB, GL_FLOAT, colour);
		glWindowPos3fv(win);		//set the window position of the occuded pixel
		glDrawPixels(1, 1, GL_RGB, GL_FLOAT, colour);
        }
        results.close();

	glFlush();

	cout << "done";
	glutSwapBuffers();
//	getchar();	//breakpoint

}

GLfloat arrayMax(GLfloat array[], GLuint numElements)
{
        GLfloat max = 0;
        GLuint m;
        for (m = 0; m < numElements; m++){
                if (array[m] > max){
                        max = array[m];
                }
        }
        return max;
}

GLfloat arrayMin(GLfloat array[], GLuint numElements)
{
        GLfloat min = sizeof(GLfloat);
        GLuint n;
        for (n = 0; n < numElements; n++){
                if (array[n] < min){
                        min = array[n];
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

	//Callback functions:
	glutDisplayFunc(display);
	glutMainLoop();

	glmDelete(model);

	return 0;
}

