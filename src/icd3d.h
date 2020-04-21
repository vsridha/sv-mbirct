
#ifndef _ICD3D_H_
#define _ICD3D_H_

/*
float ICDStep3D(
	struct ReconParams reconparams,
	float THETA1,
	float THETA2,
	float tempV,
	float *neighbors);
*/


void PandP_Update(
	float SigmaPnPsq,
	float tempV,
	float tempProxMap,
	float *THETA1,
	float *THETA2);

void QGGMRF3D_Update(
	struct PriorParams priorparams,
	float tempV,
	float *neighbors,
	float *THETA1,
	float *THETA2);

float QGGMRF_SurrogateCoeff(
	float delta,
	struct PriorParams priorparams);

float QGGMRF_Potential(float delta, struct PriorParams priorparams);
void ExtractNeighbors_WithinSlice(float *neighbors,int jx,int jy, float *image,struct ImageParams3D imgparams);

#endif
