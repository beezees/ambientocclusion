//coordinates.c
//Description: Converts screen space coordinates to object space, then to voxel space.
#include <GL/gl.h>
#include <GL/glu.h>
#include <math.h>

//Window dimensions:
float winWidth = 500;
float winHeight = 500;

//Number of voxel grid divisions:
float grid = 256;

double convertToObj(int x, int y)
{
    int viewport[4];
    double modelview[16];
    double projection[16];
    float z;
	double objx, objy, objz;
	double obj[3];

    glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
    glGetDoublev( GL_PROJECTION_MATRIX, projection );
    glGetIntegerv( GL_VIEWPORT, viewport );

    glReadPixels( x, viewport[3]-y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z);

    gluUnProject( x, viewport[3]-y, z, modelview, projection, viewport, &objx, &objy, &objz);
    //printf("In world coords (x,y,z) = %f %f %f \n", objx, objy, objz);
	
	obj[0] = objx;
	obj[1] = objy;
	obj[2] = objz;

	return obj;
}

double convertToVoxel(double objx, double objy, double objz)
{
	double vox[3];
	double Vx, Vy, Vz;
	double MaxX[3], MaxY[3], MaxZ[3];
	double objxMax, objyMax, objzMax;
	double voxSizeX, voxSizeY, voxSizeZ;

	//Find maximum dimensions of object space:
	MaxX = convertToObj(winWidth, 0);
	objxMax = MaxX[0];
	MaxY = convertToObj(0, winHeight);
	objyMax = MaxY[1];
	objzMax = objyMax;
	
	//Find dimensions of a single voxel:
	voxSizeX = objxMax/grid;
	voxSizeY = objyMax/grid;
	voxSizeZ = objzMax/grid;

	//Coordinates of containing voxel:
	vox[0] = floor(objx/voxSizeX);
	vox[1] = floor(objy/voxSizeY);
	vox[2] = floor(objz/voxSizeZ);

    return vox; 
}

int main (int argc, char **argv)
{
	//Screen space values to be converted:
	int x = 300;
	int y = 592;
	double obj[3];
	double vox[3];

	obj = convertToObj(x, y);
	vox = convertToVoxel(obj[0], obj[1] , obj[2]);
	printf("Voxel coordinates (x,y,z) = %f %f %f \n", vox[0], vox[1], vox[2]);

	return 0;
}
