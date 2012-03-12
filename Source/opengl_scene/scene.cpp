//scene.cpp
//Calina Copos & Afton Geil
//EEC277 Final Project
#include <GL/glut.h>
#include <math.h>
#include "include/glm/glm.h"
#include <stdio.h>
#include <iostream>

int winWidth = 800;
int winHeight = 600;

int displayList;

GLMmodel* model;

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
    printf("In world coords (x,y,z) = %f %f %f \n", objx, objy, objz);

    obj[0] = objx;
    obj[1] = objy;
    obj[2] = objz;

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
	std::cout << "Min x:";
	std::cout << xObjMin << "\n";
	std::cout << "Min y:";
	std::cout << yObjMin << "\n";
	std::cout << "Min z:";
	std::cout << zObjMin << "\n";
	std::cout << "Max x:";
	std::cout << xObjMax << "\n";
	std::cout << "Max y:";
	std::cout << yObjMax << "\n";
	std::cout << "Max z:";
	std::cout << zObjMax << "\n";

	//Display model:
	displayList = glGenLists(1);
	glNewList(displayList, GL_COMPILE);
		glColor4f(1.0, 0.0, 1.0, 1.0);
		glmList(model, GLM_NONE);
	glEndList();

	//TODO: Convert from window location to closest voxel vertex location:
	int pixelX, pixelY;
	pixelX = 650;
	pixelY = 547;
	double obj[3];
	double objx, objy, objz;
	convertToObj(pixelX, pixelY, obj);
	objx = obj[0];
	objy = obj[1];
	objz = obj[2];
	std::cout << "Object Space Values:";
	std::cout << objx << "\n" << objy << "\n" << objz <<"\n";

	//Callback functions:
	glutDisplayFunc(display);
	glutMainLoop();

	glmDelete(model);

	return 0;
}

