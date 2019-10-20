#ifndef _MBIRCT_H_
#define _MBIRCT_H_


//#define USE_INTEL_MEMCPY
//#define find_RMSE 

#define SVLength 9
#define PIECELEN 48
#define overlappingDistance1 2
#define overlappingDistance2 2
#define SV_depth 4

#if 0
// Experimental
struct SVParam
{
	struct minStruct *bandMinMap;
	struct maxStruct *bandMaxMap;
	int Nsv;
	int SVLength;
	int SV_sidelen;
	int SV_Nxy;
	int SV_depth;
	int pieceLength;
	int overlap;
};

// Experimental
struct SysMatrixSV
{
	struct AValues_char **A_Padded_Map;
	float *max_num_pointer;
};
#endif


struct CmdLine
{
    char SinoParamsFile[256];
    char ImageParamsFile[256];
    char ReconParamsFile[256];
    char SinoDataFile[256];
    char SinoWeightsFile[256];
    char ReconImageDataFile[256]; /* output */
    char SysMatrixFile[256];
    char InitImageDataFile[256]; /* optional input */
};



#endif
