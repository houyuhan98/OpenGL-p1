// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <array>
#include <sstream>
#include <math.h>
// Include GLEW
#include <GL/glew.h>
// Include GLFW
#include <GLFW/glfw3.h>
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
using namespace glm;
// Include AntTweakBar
#include <AntTweakBar.h>
#include <common/shader.hpp>
#include <common/controls.hpp>
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>

#define PI 3.1415926535897

typedef struct Vertex {
	float XYZW[4];
	float RGBA[4];
	void SetCoords(float *coords) {
		XYZW[0] = coords[0];
		XYZW[1] = coords[1];
		XYZW[2] = coords[2];
		XYZW[3] = coords[3];
	}
	void SetColor(float *color) {
		RGBA[0] = color[0];
		RGBA[1] = color[1];
		RGBA[2] = color[2];
		RGBA[3] = color[3];
	}
};

// ATTN: USE POINT STRUCTS FOR EASIER COMPUTATIONS
typedef struct point {
	float x, y, z;
	point(const float x = 0, const float y = 0, const float z = 0) : x(x), y(y), z(z){};
	point(float *coords) : x(coords[0]), y(coords[1]), z(coords[2]){};
	point operator -(const point& a)const {
		return point(x - a.x, y - a.y, z - a.z);
	}
	point operator +(const point& a)const {
		return point(x + a.x, y + a.y, z + a.z);
	}
	point operator *(const float& a)const {
		return point(x*a, y*a, z*a);
	}
	point operator /(const float& a)const {
		return point(x / a, y / a, z / a);
	}
	float* toArray() {
		float array[] = { x, y, z, 1.0f };
		return array;
	}
};

// function prototypes
int initWindow(void);
void initOpenGL(void);
void createVAOs(Vertex[], unsigned short[], size_t, size_t, int);
void createObjects(void);
void pickVertex(void);
void moveVertex(void);
void drawScene(void);
void cleanup(void);

static void mouseCallback(GLFWwindow*, int, int, int);
static void keyCallback(GLFWwindow*, int, int, int, int);

// Subdivision
void Subdivision(Vertex*, const Vertex*, int);
// Bezier Curves
void BezierCurves(const Vertex*, Vertex*);
// Catmull-Rom Curves
void CatmullRomPts(const Vertex*, Vertex*);
void CatmullRomCurves(const Vertex*, Vertex*);

// GLOBAL VARIABLES
GLFWwindow* window;
const GLuint window_width = 1024, window_height = 768;

glm::mat4 gProjectionMatrix;
glm::mat4 gViewMatrix;

GLuint gPickedIndex;
std::string gMessage;

GLuint programID;
GLuint pickingProgramID;

int pressed = 0;
float pickedR;
float pickedG;
float pickedB;
bool isChanged = false;
int count = 0;
int curloopPos = 0;
bool zPick = false;
bool splitView = false;
bool loop = false;

// ATTN: INCREASE THIS NUMBER AS YOU CREATE NEW OBJECTS
const GLuint NumObjects = 10;	// number of different "objects" to be drawn
GLuint VertexArrayId[NumObjects] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
GLuint VertexBufferId[NumObjects] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
GLuint IndexBufferId[NumObjects] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
size_t NumVert[NumObjects] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

GLuint MatrixID;
GLuint ViewMatrixID;
GLuint ModelMatrixID;
GLuint PickingMatrixID;
GLuint pickingColorArrayID;
GLuint pickingColorID;
GLuint LightID;

// Define objects
Vertex Vertices[] =
{
	{ { 1.0f, 0.5f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 0
	{ { 0.5f, 1.5f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 1
	{ { -0.5f, 1.5f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 2
	{ { -1.0f, 0.5f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 3
	{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 4
	{ { 1.0f, -0.5f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 5
	{ { 0.5f, -1.5f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 6
	{ { -0.5f, -1.5f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 7 
	{ { -1.0f, -0.5f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }, // 8
	{ { 0.0f, 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } } // 9 
};

Vertex* PV = Vertices;

unsigned short Indices[] = {
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9
};

const size_t IndexCount = sizeof(Indices) / sizeof(unsigned short);
// ATTN: DON'T FORGET TO INCREASE THE ARRAY SIZE IN THE PICKING VERTEX SHADER WHEN YOU ADD MORE PICKING COLORS
float pickingColor[IndexCount] = { 0 / 255.0f, 1 / 255.0f, 2 / 255.0f, 3 / 255.0f, 4 / 255.0f, 5 / 255.0f, 6 / 255.0f, 7 / 255.0f, 8 / 255.0f, 9 / 255.0f};
unsigned short MC = IndexCount; // store a copy of originl indexcount
std::vector<int> subIndexCount; // a int vector to store different indexcount for each subdivision level

// ATTN: ADD YOU PER-OBJECT GLOBAL ARRAY DEFINITIONS HERE
// indices array for each subdivision level
unsigned short level1[20];
unsigned short level2[40];
unsigned short level3[80];
unsigned short level4[160];
unsigned short level5[320];
// vertex array for each level of subdivision
Vertex subdivision1[20];
Vertex subdivision2[40];
Vertex subdivision3[80];
Vertex subdivision4[160];
Vertex subdivision5[320];
Vertex* PS1 = subdivision1;
Vertex* PS2 = subdivision2;
Vertex* PS3 = subdivision3;
Vertex* PS4 = subdivision4;
Vertex* PS5 = subdivision5;

// bezier curves indices
unsigned short bezierIndices[40];
// bezier curves vertex array
Vertex beziercurve[40];
Vertex* PB = beziercurve;

unsigned short catmullromIndices[40]; // indices for bezier points
unsigned short decastelIndices[150]; // indices for CR curve
// CR vertex array
Vertex catmullrom[40];
Vertex decastel[150]; // 15 points per segment
Vertex* PCR = catmullrom;
Vertex* PD = decastel;

// looping dot's indice and vertex array
unsigned short dotloopIndex[] = {0}; 
Vertex dotloop[1];

float subdivideColor[] = { 0.0f, 1.0f, 1.0f, 1.0f }; // cyan
float bezierColor[] = { 1.0f, 1.0f, 0.0f, 1.0f }; // yellow
float CRptColor[] = { 1.0f, 0.0f, 0.0f, 1.0f }; // red
float CRcurveColor[] = { 0.0f, 1.0f, 0.0f, 1.0f }; // green
float zpickColor[] = { 0.0f, 0.0f, 1.0f, 1.0f }; // blue color for picked axis in Z plane movement
float xypickColor[] = { 1.0f, 0.0f, 0.0f, 1.0f }; // red color for picked axis in XY plane movement
float dotloopColor[] = { 1.0f, 1.0f, 0.0f, 1.0f }; // yellow color for looping vertex

void createObjects(void)
{
	// ATTN: DERIVE YOUR NEW OBJECTS HERE:
	// each has one vertices {pos;color} and one indices array (no picking needed here)
	if (pressed == 1) {
		if (count % 6 != 0) {
			Subdivision(PS1, PV, 1);
			if (count > 1) {
				Subdivision(PS2, PS1, 2);
			}
			if (count > 2) {
				Subdivision(PS3, PS2, 3);
			}
			if (count > 3) {
				Subdivision(PS4, PS3, 4);
			}
			if (count > 4) {
				Subdivision(PS5, PS4, 5);
			}
		}
		else {
			// Reset
			count = 0;
		}
	}
	if (pressed == 2) {
		BezierCurves(PV, PB);
	}
	if (pressed == 3) {
		CatmullRomPts(PV,PCR);
		CatmullRomCurves(PCR,PD);
	}
	if (loop) {//update dot postion to run on catmull rom curve
		if (curloopPos % 150 == 0) {
			curloopPos = 0;
		}
		CatmullRomPts(PV, PCR);
		CatmullRomCurves(PCR, PD);
		dotloop[0].SetCoords(decastel[curloopPos].XYZW);
		dotloop[0].SetColor(dotloopColor);
		curloopPos++;
	}
}

void Subdivision(Vertex* cur, const Vertex* pre, int level) {

	float x0,x1;
	float y0,y1;
	int a,b;
	int Vnum = subIndexCount.at(level - 1);

	for (int i = 0; i < Vnum; i++) {
		a = i - 1;
		b = i + 1;
		if (i == 0) {
			a = Vnum - 1;
		}
		if (i == Vnum - 1) {
			b = 0;
		}

		// calc P2i+1 xy, set xy and color
		x0 = (pre[a].XYZW[0] + (6 * pre[i].XYZW[0]) + pre[b].XYZW[0]) / 8;
		y0 = (pre[a].XYZW[1] + (6 * pre[i].XYZW[1]) + pre[b].XYZW[1]) / 8;
		float newCoords0[] = { x0, y0, 0.0f, 1.0f };
		cur[(i * 2) + 1].SetCoords(newCoords0);
		cur[(i * 2) + 1].SetColor(subdivideColor);
		// calc P2i xy, set xy and color
		x1 = ((4 * pre[a].XYZW[0]) + (4 * pre[i].XYZW[0])) / 8;
		y1 = ((4 * pre[a].XYZW[1]) + (4 * pre[i].XYZW[1])) / 8;
		float newCoords1[] = { x1, y1, 0.0f, 1.0f };
		cur[i * 2].SetCoords(newCoords1);
		cur[i * 2].SetColor(subdivideColor);
	}

}

void BezierCurves(const Vertex* p, Vertex* c) {

	float x0,x1,x2,x3; 
	float y0,y1,y2,y3;
	int a, b, e;

	for (int i = 0; i < IndexCount; i++) {
		a = i + 1;
		b = i - 1;
		e = i + 2;
		if (i == IndexCount - 1) {
			a = 0;
		}
		if (i == 0) {
			b = IndexCount - 1;
		}
		if (i == IndexCount - 2) {
			e = 0;
		}

		// calc c1 xy
		x1 = ((2 * p[i].XYZW[0]) + p[a].XYZW[0]) / 3;
		y1 = ((2 * p[i].XYZW[1]) + p[a].XYZW[1]) / 3;
		float newCoords1[] = { x1, y1, 0.0f, 1.0f };
		// calc c2 xy
		x2 = (p[i].XYZW[0] + (2 * p[a].XYZW[0])) / 3;
		y2 = (p[i].XYZW[1] + (2 * p[a].XYZW[1])) / 3;
		float newCoords2[] = { x2, y2, 0.0f, 1.0f };

		// Set xy and color for c1
		c[4 * i + 1].SetCoords(newCoords1);
		c[4 * i + 1].SetColor(bezierColor);
		// Set xy and color for c2
		c[4 * i + 2].SetCoords(newCoords2);
		c[4 * i + 2].SetColor(bezierColor);

		// calc c0 xy
		x0 = (p[b].XYZW[0] + (2 * p[i].XYZW[0])) / 3;
		y0 = (p[b].XYZW[1] + (2 * p[i].XYZW[1])) / 3;
		// midpoint
		x0 = (x0 + x1) / 2;
		y0 = (y0 + y1) / 2;
		float newCoords0[] = { x0, y0, 0.0f, 1.0f };
		// calc c3 xy
		x3 = ((2 * p[a].XYZW[0]) + p[e].XYZW[0]) / 3;
		y3 = ((2 * p[a].XYZW[1]) + p[e].XYZW[1]) / 3;
		// midpoint
		x3 = (x3 + x2) / 2;
		y3 = (y3 + y2) / 2;
		float newCoords3[] = { x3, y3, 0.0f, 1.0f };

		// Set xy and color for c0
		c[4 * i].SetCoords(newCoords0);
		c[4 * i].SetColor(bezierColor);
		// Set xy and color for c3
		c[4 * i + 3].SetCoords(newCoords3);
		c[4 * i + 3].SetColor(bezierColor);
	}
}

void CatmullRomPts(const Vertex* p, Vertex* c) {

	float x0, x1, x2, x3;
	float y0, y1, y2, y3;
	int a, b, e;

	for (int i = 0; i < IndexCount; i++) {
		if (i == IndexCount - 1) {
			a = 0;
		}
		else {
			a = i + 1;
		}
		if (a == IndexCount - 1) {
			e = 0;
		}
		else {
			e = a + 1;
		}
		if (i == 0) {
			b = IndexCount - 1;
		}
		else {
			b = i - 1;
		}

		float w = 0.2; // t value
		float x1T = w * (p[a].XYZW[0] - p[b].XYZW[0]); // x1 tangent, ci0
		float y1T = w * (p[a].XYZW[1] - p[b].XYZW[1]); // y1 tangent
		float x2T = w * (p[e].XYZW[0] - p[i].XYZW[0]); // x2 tangent, ci3
		float y2T = w * (p[e].XYZW[1] - p[i].XYZW[1]); // y2 tangent

		// Calc c0,c1,c2,c3 xy
		x0 = p[i].XYZW[0];
		y0 = p[i].XYZW[1];
		x3 = p[a].XYZW[0];
		y3 = p[a].XYZW[1];
		x1 = x0 + x1T;
		y1 = y0 + y1T;
		x2 = x3 - x2T;
		y2 = y3 - y2T;

		// Set xy and color for c0,c1,c2,c3
		float newCoords0[] = { x0, y0, 0.0f, 1.0f };
		c[4 * i].SetCoords(newCoords0);
		c[4 * i].SetColor(CRptColor);
		float newCoords1[] = { x1, y1, 0.0f, 1.0f };
		c[4 * i + 1].SetCoords(newCoords1);
		c[4 * i + 1].SetColor(CRptColor);
		float newCoords2[] = { x2, y2, 0.0f, 1.0f };
		c[4 * i + 2].SetCoords(newCoords2);
		c[4 * i + 2].SetColor(CRptColor);
		float newCoords3[] = { x3, y3, 0.0f, 1.0f };
		c[4 * i + 3].SetCoords(newCoords3);
		c[4 * i + 3].SetColor(CRptColor);
	}
}

void CatmullRomCurves(const Vertex* pcr, Vertex* curve) {
	
	Vertex temp[40]; // temporary array to hold data from calc
	float x, y, t;

	for (int i = 0; i < IndexCount; i++) {
		for (int j = 0; j < 15; j++) {
			// copy bezier points from PCR into temp array
			for (int k = 0; k < 40; k++) {
				temp[k] = pcr[k];
			}
			// calc curve using decasteljau
			for (int a = 1; a < 4; a++) {
				for (int b = 0; b < 4 - a; b++) {
					// calc x,y
					t = j / 15.0f;
					x = (1.0f - t) * temp[4 * i + b].XYZW[0] + t * temp[4 * i + b + 1].XYZW[0];	
					y = (1.0f - t) * temp[4 * i + b].XYZW[1] + t * temp[4 * i + b + 1].XYZW[1];	
					// set xy and color for curve
					float newCoords[] = { x, y, 0.0f, 1.0f };
					temp[4 * i + b].SetCoords(newCoords);
					temp[4 * i + b].SetColor(CRcurveColor);	
				}
			}
			// copy data from temp array to PD VAO
			curve[(15 * i) + j].SetCoords(temp[4 * i].XYZW);
			curve[(15 * i) + j].SetColor(CRcurveColor);
		}
	}
}

void drawScene(void)
{
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
	// Re-clear the screen for real rendering
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(programID);
	{
		glm::mat4 ModelMatrix = glm::mat4(1.0); // TranslationMatrix * RotationMatrix;
		glm::mat4 ModelMatrix2 = glm::mat4(1.0);  // Second ModelMatrix for Translation
		glm::mat4 MVP;
		glm::mat4 MVP2;
		if (splitView) {
			// Scale ModelMatrix
			ModelMatrix = glm::scale(ModelMatrix,glm::vec3(0.8f));
			// Translate ModelMatrix2 Downwards from Origin
			ModelMatrix2 = glm::translate(ModelMatrix,glm::vec3(0.0f, -2.0f, 0.0f));
		    // Translate ModelMatrix Upwards from Origin
			ModelMatrix = glm::translate(ModelMatrix,glm::vec3(0.0f, 2.0f, 0.0f));
		    // Rotate ModelMatrix2 Around Y Axis by PI/2 for side view
			ModelMatrix2 = glm::rotate(ModelMatrix2,float(PI / 2),glm::vec3(0.0f, 1.0f, 0.0f));
		}
		MVP = gProjectionMatrix * gViewMatrix * ModelMatrix;
		MVP2 = gProjectionMatrix * gViewMatrix * ModelMatrix2;

		// Send our transformation to the currently bound shader, 
		// in the "MVP" uniform
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
		glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);
		glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &gViewMatrix[0][0]);
		glm::vec3 lightPos = glm::vec3(4, 4, 4);
		glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);
		
		glEnable(GL_PROGRAM_POINT_SIZE);

		glBindVertexArray(VertexArrayId[0]);	// draw Vertices
		glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[0]);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertices), Vertices);				// update buffer data
		glDrawElements(GL_LINE_LOOP, NumVert[0], GL_UNSIGNED_SHORT, (void*)0);
		glDrawElements(GL_POINTS, NumVert[0], GL_UNSIGNED_SHORT, (void*)0);
		// ATTN: OTHER BINDING AND DRAWING COMMANDS GO HERE, one set per object:
		//glBindVertexArray(VertexArrayId[<x>]); etc etc
		if (pressed == 1) {
			if (count == 1) {
				glBindVertexArray(VertexArrayId[1]);
				glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[1]);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision1), subdivision1);
				glDrawElements(GL_LINE_LOOP, NumVert[1], GL_UNSIGNED_SHORT, (void*)0);
				glDrawElements(GL_POINTS, NumVert[1], GL_UNSIGNED_SHORT, (void*)0);
				glBindVertexArray(0);
			}
			if (count == 2) {
				glBindVertexArray(VertexArrayId[2]);
				glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[2]);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision2), subdivision2);
				glDrawElements(GL_LINE_LOOP, NumVert[2], GL_UNSIGNED_SHORT, (void*)0);
				glDrawElements(GL_POINTS, NumVert[2], GL_UNSIGNED_SHORT, (void*)0);
				glBindVertexArray(0);
			}
			if (count == 3) {
				glBindVertexArray(VertexArrayId[3]);
				glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[3]);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision3), subdivision3);
				glDrawElements(GL_LINE_LOOP, NumVert[3], GL_UNSIGNED_SHORT, (void*)0);
				glDrawElements(GL_POINTS, NumVert[3], GL_UNSIGNED_SHORT, (void*)0);
				glBindVertexArray(0);
			}
			if (count == 4) {
				glBindVertexArray(VertexArrayId[4]);
				glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[4]);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision4), subdivision4);
				glDrawElements(GL_LINE_LOOP, NumVert[4], GL_UNSIGNED_SHORT, (void*)0);
				glDrawElements(GL_POINTS, NumVert[4], GL_UNSIGNED_SHORT, (void*)0);
				glBindVertexArray(0);
			}
			if (count == 5) {
				glBindVertexArray(VertexArrayId[5]);
				glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[5]);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision5), subdivision5);
				glDrawElements(GL_LINE_LOOP, NumVert[5], GL_UNSIGNED_SHORT, (void*)0);
				glDrawElements(GL_POINTS, NumVert[5], GL_UNSIGNED_SHORT, (void*)0);
				glBindVertexArray(0);
			}
		}
		if (pressed == 2) {
			glBindVertexArray(VertexArrayId[6]);
			glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[6]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(beziercurve), beziercurve);
			glDrawElements(GL_LINE_LOOP, NumVert[6], GL_UNSIGNED_SHORT, (void*)0);
			glDrawElements(GL_POINTS, NumVert[6], GL_UNSIGNED_SHORT, (void*)0);
			glBindVertexArray(0);
		}
		if (pressed == 3) {
			glBindVertexArray(VertexArrayId[7]);
			glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[7]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(catmullrom), catmullrom);
			glDrawElements(GL_LINE_LOOP, NumVert[7], GL_UNSIGNED_SHORT, (void*)0);
			glDrawElements(GL_POINTS, NumVert[7], GL_UNSIGNED_SHORT, (void*)0);
			glBindVertexArray(0);

			glBindVertexArray(VertexArrayId[8]);
			glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[8]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(decastel), decastel);
			glDrawElements(GL_LINE_LOOP, NumVert[8], GL_UNSIGNED_SHORT, (void*)0);
			glBindVertexArray(0);
		}
		if (loop) {
			glBindVertexArray(VertexArrayId[9]);
			glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[9]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(dotloop), dotloop);
			glDrawElements(GL_POINTS, NumVert[9], GL_UNSIGNED_SHORT, (void*)0);
		}
		glBindVertexArray(0);

		if (splitView) {
			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP2[0][0]);
			glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix2[0][0]);
			glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &gViewMatrix[0][0]);
			glm::vec3 lightPos = glm::vec3(4, 4, 4);
			glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

			glEnable(GL_PROGRAM_POINT_SIZE);

			glBindVertexArray(VertexArrayId[0]);	// draw Vertices
			glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[0]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertices), Vertices);
			glDrawElements(GL_LINE_LOOP, NumVert[0], GL_UNSIGNED_SHORT, (void*)0);
			glDrawElements(GL_POINTS, NumVert[0], GL_UNSIGNED_SHORT, (void*)0);

			if (pressed == 1) {
				if (count == 1) {
					glBindVertexArray(VertexArrayId[1]);
					glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[1]);
					glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision1), subdivision1);
					glDrawElements(GL_LINE_LOOP, NumVert[1], GL_UNSIGNED_SHORT, (void*)0);
					glDrawElements(GL_POINTS, NumVert[1], GL_UNSIGNED_SHORT, (void*)0);
					glBindVertexArray(0);
				}
				if (count == 2) {
					glBindVertexArray(VertexArrayId[2]);
					glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[2]);
					glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision2), subdivision2);
					glDrawElements(GL_LINE_LOOP, NumVert[2], GL_UNSIGNED_SHORT, (void*)0);
					glDrawElements(GL_POINTS, NumVert[2], GL_UNSIGNED_SHORT, (void*)0);
					glBindVertexArray(0);
				}
				if (count == 3) {
					glBindVertexArray(VertexArrayId[3]);
					glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[3]);
					glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision3), subdivision3);
					glDrawElements(GL_LINE_LOOP, NumVert[3], GL_UNSIGNED_SHORT, (void*)0);
					glDrawElements(GL_POINTS, NumVert[3], GL_UNSIGNED_SHORT, (void*)0);
					glBindVertexArray(0);
				}
				if (count == 4) {
					glBindVertexArray(VertexArrayId[4]);
					glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[4]);
					glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision4), subdivision4);
					glDrawElements(GL_LINE_LOOP, NumVert[4], GL_UNSIGNED_SHORT, (void*)0);
					glDrawElements(GL_POINTS, NumVert[4], GL_UNSIGNED_SHORT, (void*)0);
					glBindVertexArray(0);
				}
				if (count == 5) {
					glBindVertexArray(VertexArrayId[5]);
					glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[5]);
					glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subdivision5), subdivision5);
					glDrawElements(GL_LINE_LOOP, NumVert[5], GL_UNSIGNED_SHORT, (void*)0);
					glDrawElements(GL_POINTS, NumVert[5], GL_UNSIGNED_SHORT, (void*)0);
					glBindVertexArray(0);
				}
			}
			if (pressed == 2) {
				glBindVertexArray(VertexArrayId[6]);
				glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[6]);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(beziercurve), beziercurve);
				glDrawElements(GL_LINE_LOOP, NumVert[6], GL_UNSIGNED_SHORT, (void*)0);
				glDrawElements(GL_POINTS, NumVert[6], GL_UNSIGNED_SHORT, (void*)0);
				glBindVertexArray(0);
			}
			if (pressed == 3) {
				glBindVertexArray(VertexArrayId[7]);
				glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[7]);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(catmullrom), catmullrom);
				glDrawElements(GL_LINE_LOOP, NumVert[7], GL_UNSIGNED_SHORT, (void*)0);
				glDrawElements(GL_POINTS, NumVert[7], GL_UNSIGNED_SHORT, (void*)0);
				glBindVertexArray(0);

				glBindVertexArray(VertexArrayId[8]);
				glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[8]);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(decastel), decastel);
				glDrawElements(GL_LINE_LOOP, NumVert[8], GL_UNSIGNED_SHORT, (void*)0);
				glBindVertexArray(0);
			}
			if (loop) {
				glBindVertexArray(VertexArrayId[9]);
				glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[9]);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(dotloop), dotloop);
				glDrawElements(GL_POINTS, NumVert[9], GL_UNSIGNED_SHORT, (void*)0);
			}
			glBindVertexArray(0);
		}
	}
	glUseProgram(0);
	// Draw GUI
	TwDraw();

	// Swap buffers
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void pickVertex(void)
{
	// Clear the screen in white
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(pickingProgramID);
	{
		glm::mat4 ModelMatrix = glm::mat4(1.0); // TranslationMatrix * RotationMatrix;
		if (splitView) {
			// Scale Matrix
			ModelMatrix = glm::scale(ModelMatrix,glm::vec3(0.8f));
			// Translate Matrix
			ModelMatrix = glm::translate(ModelMatrix,glm::vec3(0.0f, 2.0f, 0.0f));
		}
		glm::mat4 MVP = gProjectionMatrix * gViewMatrix * ModelMatrix;

		// Send our transformation to the currently bound shader, in the "MVP" uniform
		glUniformMatrix4fv(PickingMatrixID, 1, GL_FALSE, &MVP[0][0]);
		glUniform1fv(pickingColorArrayID, NumVert[0], pickingColor);	// here we pass in the picking marker array

		// Draw the ponts
		glEnable(GL_PROGRAM_POINT_SIZE);
		glBindVertexArray(VertexArrayId[0]);
		glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[0]);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Vertices), Vertices);	// update buffer data
		glDrawElements(GL_POINTS, NumVert[0], GL_UNSIGNED_SHORT, (void*)0);
		glBindVertexArray(0);
	}
	glUseProgram(0);
	// Wait until all the pending drawing commands are really done.
	// Ultra-mega-over slow ! 
	// There are usually a long time between glDrawElements() and
	// all the fragments completely rasterized.
	glFlush();
	glFinish();

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Read the pixel at the center of the screen.
	// You can also use glfwGetMousePos().
	// Ultra-mega-over slow too, even for 1 pixel, 
	// because the framebuffer is on the GPU.
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	unsigned char data[4];
	glReadPixels(xpos, window_height - ypos, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, data); // OpenGL renders with (0,0) on bottom, mouse reports with (0,0) on top

	// Convert the color back to an integer ID
	if (!isChanged) {
		gPickedIndex = int(data[0]);
	}
	if (gPickedIndex == 255) { // Full white, must be the background !
		gMessage = "background";
	}
	else {
		if (!isChanged && gPickedIndex < IndexCount) {
			pickedR = Vertices[gPickedIndex].RGBA[0];
			pickedG = Vertices[gPickedIndex].RGBA[1];
			pickedB = Vertices[gPickedIndex].RGBA[2];
		}
		isChanged = true;

	}
	// Uncomment these lines to see the picking shader in effect
	//glfwSwapBuffers(window);
	//continue; // skips the normal rendering
}

// fill this function in!
void moveVertex(void)
{
	glm::mat4 ModelMatrix = glm::mat4(1.0);
	if (splitView) {
		// Scale Matrix
		ModelMatrix = glm::scale(ModelMatrix,glm::vec3(0.8f));
		// Translate Matrix
		ModelMatrix = glm::translate(ModelMatrix,glm::vec3(0.0f, 2.0f, 0.0f));
	}

	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	glm::vec4 vp = glm::vec4(viewport[0], viewport[1], viewport[2], viewport[3]);
	// retrieve your cursor position
	// get your world coordinates
	// move points
	if (isChanged) {
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		glm::vec3 mouseLoc = glm::unProject(glm::vec3(window_width - xpos, window_height - ypos, 0.0), ModelMatrix, gProjectionMatrix, vp);
		if (!zPick) {
			if (gPickedIndex < IndexCount) {
				Vertices[gPickedIndex].XYZW[0] = mouseLoc[0];
				Vertices[gPickedIndex].XYZW[1] = mouseLoc[1];
				Vertices[gPickedIndex].SetColor(xypickColor);
			}
		}
		else {
			if (gPickedIndex < IndexCount) {
				Vertices[gPickedIndex].XYZW[2] = mouseLoc[1]; // z translate when mouse moves up and down
				Vertices[gPickedIndex].SetColor(zpickColor);
			}
		}
		
	}

	
	if (gPickedIndex == 255){ // Full white, must be the background !
		gMessage = "background";
	}
	else {
		std::ostringstream oss;
		oss << "point " << gPickedIndex;
		gMessage = oss.str();
	}
}

int initWindow(void)
{
	// Initialise GLFW
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // FOR MAC

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(window_width, window_height, "Hou Yuhan(20199280)", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	// Initialize the GUI
	TwInit(TW_OPENGL_CORE, NULL);
	TwWindowSize(window_width, window_height);
	TwBar * GUI = TwNewBar("Picking");
	TwSetParam(GUI, NULL, "refresh", TW_PARAM_CSTRING, 1, "0.1");
	TwAddVarRW(GUI, "Last picked object", TW_TYPE_STDSTRING, &gMessage, NULL);

	// Set up inputs
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_FALSE);
	glfwSetCursorPos(window, window_width / 2, window_height / 2);
	glfwSetMouseButtonCallback(window, mouseCallback);
	glfwSetKeyCallback(window, keyCallback);

	return 0;
}

void initOpenGL(void)
{
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	//glm::mat4 ProjectionMatrix = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 100.0f);
	// Or, for an ortho camera :
	gProjectionMatrix = glm::ortho(-4.0f, 4.0f, -3.0f, 3.0f, 0.0f, 100.0f); // In world coordinates

	// Camera matrix
	gViewMatrix = glm::lookAt(
		glm::vec3(0, 0, -5), // Camera is at (4,3,3), in World Space
		glm::vec3(0, 0, 0), // and looks at the origin
		glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
		);

	// Create and compile our GLSL program from the shaders
	programID = LoadShaders("hw1bShade.vertexshader", "hw1bShade.fragmentshader");
	pickingProgramID = LoadShaders("hw1bPick.vertexshader", "hw1bPick.fragmentshader");

	// Get a handle for our "MVP" uniform
	MatrixID = glGetUniformLocation(programID, "MVP");
	ViewMatrixID = glGetUniformLocation(programID, "V");
	ModelMatrixID = glGetUniformLocation(programID, "M");
	PickingMatrixID = glGetUniformLocation(pickingProgramID, "MVP");
	// Get a handle for our "pickingColorID" uniform
	pickingColorArrayID = glGetUniformLocation(pickingProgramID, "PickingColorArray");
	pickingColorID = glGetUniformLocation(pickingProgramID, "PickingColor");
	// Get a handle for our "LightPosition" uniform
	LightID = glGetUniformLocation(programID, "LightPosition_worldspace");

	createVAOs(Vertices, Indices, sizeof(Vertices), sizeof(Indices), 0);
	// Subdivision VAOs
	createVAOs(subdivision1, level1, sizeof(subdivision1), sizeof(level1), 1);
	createVAOs(subdivision2, level2, sizeof(subdivision2), sizeof(level2), 2);
	createVAOs(subdivision3, level3, sizeof(subdivision3), sizeof(level3), 3);
	createVAOs(subdivision4, level4, sizeof(subdivision4), sizeof(level4), 4);
	createVAOs(subdivision5, level5, sizeof(subdivision5), sizeof(level5), 5);
	// Bezier Curves VAO
	createVAOs(beziercurve, bezierIndices, sizeof(beziercurve), sizeof(bezierIndices), 6);
	// Catmull-Rom Curves VAOs
	createVAOs(catmullrom, catmullromIndices, sizeof(catmullrom), sizeof(catmullromIndices), 7);
	createVAOs(decastel, decastelIndices, sizeof(decastel), sizeof(decastelIndices), 8);
	// Looping vertex VAO
	createVAOs(dotloop, dotloopIndex, sizeof(dotloop), sizeof(dotloopIndex), 9);

	createObjects();

	// ATTN: create VAOs for each of the newly created objects here:
	// createVAOs(<fill this appropriately>);

}

void createVAOs(Vertex Vertices[], unsigned short Indices[], size_t BufferSize, size_t IdxBufferSize, int ObjectId) {

	NumVert[ObjectId] = IdxBufferSize / (sizeof GLubyte);

	GLenum ErrorCheckValue = glGetError();
	size_t VertexSize = sizeof(Vertices[0]);
	size_t RgbOffset = sizeof(Vertices[0].XYZW);

	// Create Vertex Array Object
	glGenVertexArrays(1, &VertexArrayId[ObjectId]);
	glBindVertexArray(VertexArrayId[ObjectId]);

	// Create Buffer for vertex data
	glGenBuffers(1, &VertexBufferId[ObjectId]);
	glBindBuffer(GL_ARRAY_BUFFER, VertexBufferId[ObjectId]);
	glBufferData(GL_ARRAY_BUFFER, BufferSize, Vertices, GL_STATIC_DRAW);

	// Create Buffer for indices
	glGenBuffers(1, &IndexBufferId[ObjectId]);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[ObjectId]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, IdxBufferSize, Indices, GL_STATIC_DRAW);

	// Assign vertex attributes
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, VertexSize, 0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, VertexSize, (GLvoid*)RgbOffset);

	glEnableVertexAttribArray(0);	// position
	glEnableVertexAttribArray(1);	// color

	// Disable our Vertex Buffer Object 
	glBindVertexArray(0);

	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR)
	{
		fprintf(
			stderr,
			"ERROR: Could not create a VBO: %s \n",
			gluErrorString(ErrorCheckValue)
			);
	}
}

void cleanup(void)
{
	// Cleanup VBO and shader
	for (int i = 0; i < NumObjects; i++) {
		glDeleteBuffers(1, &VertexBufferId[i]);
		glDeleteBuffers(1, &IndexBufferId[i]);
		glDeleteVertexArrays(1, &VertexArrayId[i]);
	}
	glDeleteProgram(programID);
	glDeleteProgram(pickingProgramID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}

static void mouseCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		pickVertex();
	}
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
		if (isChanged && gPickedIndex < IndexCount) {
			Vertices[gPickedIndex].RGBA[0] = pickedR;
			Vertices[gPickedIndex].RGBA[1] = pickedG;
			Vertices[gPickedIndex].RGBA[2] = pickedB;
			isChanged = false;
		}
	}
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_1 && action == GLFW_RELEASE) {
		pressed = 1;
		count++;		
	}
	if (key == GLFW_KEY_2 && action == GLFW_RELEASE) {
		pressed = 2;
		count = 0;
	}
	if (key == GLFW_KEY_3 && action == GLFW_RELEASE) {
		pressed = 3;
		count = 0;
	}
	if (key == GLFW_KEY_4 && action == GLFW_PRESS) {
		splitView = !splitView;
	}
	if (key == GLFW_KEY_5 && action == GLFW_PRESS) {
		loop = !loop;
	}
	if ((key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) && action == GLFW_PRESS) {
		zPick = !zPick;
	}
}

int main(void)
{
	// initialize window
	int errorCode = initWindow();
	if (errorCode != 0)
		return errorCode;

	// initialize each subdivision level's index count
	subIndexCount.resize(6);  // max 5 levels then reset
	for (int i = 0; i < 6; i++) {
		if (i == 0) {
			subIndexCount.at(0) = (MC); // base is original index count
		}
		else {
			subIndexCount.at(i) = (2 * subIndexCount.at(i - 1)); // each subdivision doubles the count
		}
	}

	// initialize indices for each subdivision level
	for (int i = 0; i < 20; i++) {
		level1[i] = i;
	}
	for (int i = 0; i < 40; i++) {
		level2[i] = i;
	}
	for (int i = 0; i < 80; i++) {
		level3[i] = i;
	}
	for (int i = 0; i < 160; i++) {
		level4[i] = i;
	}
	for (int i = 0; i < 320; i++) {
		level5[i] = i;
	}

	// initialize bezier indices
	for (int i = 0; i < 40; i++) {
		bezierIndices[i] = i;
	}

	// initialize catmullrom indices
	for (int i = 0; i < 40; i++) {
		catmullromIndices[i] = i;
	}

	// initialize decasteljau indices
	for (int i = 0; i < 150; i++) {
		decastelIndices[i] = i;
	}

	// initialize OpenGL pipeline
	initOpenGL();

	// For speed computation
	double lastTime = glfwGetTime();
	int nbFrames = 0;
	do {
		// Measure speed
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lastTime >= 1.0){ // If last prinf() was more than 1sec ago
			// printf and reset
			printf("%f ms/frame\n", 1000.0 / double(nbFrames));
			nbFrames = 0;
			lastTime += 1.0;
		}

		// DRAGGING: move current (picked) vertex with cursor
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT))
			moveVertex();

		// DRAWING SCENE
		createObjects();	// re-evaluate curves in case vertices have been moved
		drawScene();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
	glfwWindowShouldClose(window) == 0);

	cleanup();

	return 0;
}
