//Calina Copos & Afton Geil
//EEC207 Project
#include <GL/glut.h>
#include <math.h>

int winWidth = 1000;
int winHeight = 750;

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
	gluLookAt(100, 100, 100, 0, 0, 0, 0, 1, 0);

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
	glColor4f(1.0, 0.0, 1.0, 1.0);
	glutSolidTeapot(20);

	glFlush();
	glutSwapBuffers();
}

int main(int argc, char** argv)
{
	//Initialize:
	glutInit(&argc, argv);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(winWidth, winHeight);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutCreateWindow("Testing Teapot");
	init();
	//Callback functions:
	glutDisplayFunc(display);
	glutMainLoop();

	return 0;
}

