#ifndef MBIR_MODULAR_DEFS_H
#define MBIR_MODULAR_DEFS_H


/* Define constants that will be used in modular MBIR framework */
#define MBIR_MODULAR_UTIL_VERSION "2.2"

#define MBIR_MODULAR_SINOTYPE_2DPARALLEL 0
#define MBIR_MODULAR_SINOTYPE_2DFAN 1	/* for future implementation */
#define MBIR_MODULAR_SINOTYPE_3DPARALLEL 2

#define MBIR_MODULAR_IMAGETYPE_2D 0
#define MBIR_MODULAR_IMAGETYPE_3D 1
#define MBIR_MODULAR_IMAGETYPE_4D 2	/* for future implementation */


#define MBIR_MAP_ESTIMATION 1
#define MBIR_PnP_PRIORS 2

#define MBIR_MODULAR_YES 1
#define MBIR_MODULAR_NO 0
#define MBIR_MODULAR_MAX_NUMBER_OF_SLICE_DIGITS 4 /* allows up to 10,000 slices */

#define PI 3.1415926535897932384
#define MUWATER 0.0202527   /* mm-1 */
#define mu2hu(Mu, MuAir, MuWater) (1000.0*(Mu-MuAir)/(MuWater-MuAir)) /* (mm^-1) to HU units conversion */
#define hu2mu(HU, MuAir, MuWater) (HU*(MuWater-MuAir)/1000.0)+MuAir   /* (mm^-1) to HU units conversion */

#define PRIOR_TYPE_QGGMRF 1
#define PRIOR_TYPE_BM3D 2
#define PRIOR_TYPE_CNN 3

/* The following utilities are used for managing data structures and files associated */
/* with the Modular MBIR Framework */

/* 3D Sinogram Parameters */
struct SinoParams3DParallel
{
    int NChannels;         /* Number of channels in detector */
    float DeltaChannel;    /* Detector spacing (mm) */
    float CenterOffset;    /* Offset of center-of-rotation ... */
                           /* Computed from center of detector in increasing direction (no. of channels) */
                           /* This can be fractional though */
    int NViews;            /* Number of view angles */
    float *ViewAngles;     /* Array of NTheta view angle entries in degrees */
    int NSlices;           /* Number of rows (slices) stored in Sino array */
    float DeltaSlice;      /* Spacing along row (slice) direction (mm) */
    int FirstSliceNumber;  /* Row (slice) index coresponding to first row (slice) stored in Sino array */
                           /* This is in absolute coordinates and is used if a partial set of slices is needed */
    int NumSliceDigits;    /* Number of slice numbers digits used in file name */
};

/* 3D Sinogram Data Structure */
struct Sino3DParallel
{
  struct SinoParams3DParallel sinoparams; /* Sinogram Parameters */
  float **sino;           /* The array is indexed by sino[Slice][ View * NChannels + Channel ] */
                          /* If data array is empty, then set Sino = NULL */
  float **weight;         /* Weights for each measurement */
};

/* 3D Image parameters*/
struct ImageParams3D
{
    int Nx;                 /* Number of columns in image */
    int Ny;                 /* Number of rows in image */
    float Deltaxy;          /* Spacing between pixels in x and y direction (mm) */
    float ROIRadius;        /* Radius of the reconstruction (mm) */
    float DeltaZ;           /* Spacing between pixels in z direction (mm) [This should be equal to DeltaSlice */
    int Nz;                 /* Number of rows (slices) in image */
    int FirstSliceNumber;   /* Detector row (slice) index cooresponding to first row (slice) stored in Image array */
                            /* This is in absolute coordinates and is used if a partial set of slices is needed */
    int NumSliceDigits;     /* Number of slice numbers digits used in file name */
};

/* 3D Image Data Structure */
struct Image3D
{
  struct ImageParams3D imgparams; /* Image parameters */
  float **image;                  /* The array is indexed by image[SliceIndex][ Row * Nx + Column ], Nx=NColumns */
                                  /* If data array is empty, then set Image = NULL */
};


struct PriorParams
{
  /*---Specific to QGGMRF---*/
  double p;               /* q-GGMRF p parameter */
  double q;               /* q-GGMRF q parameter (q=2 is typical choice) */
  double T;               /* q-GGMRF T parameter */
  double SigmaX;          /* q-GGMRF sigma_x parameter (mm-1) */
  /* neighbor weights */
  double b_nearest;       /* Relative nearest neighbor weight [default = 1] */
  double b_diag;          /* Relative diagonal neighbor weight in (x,y) plane [default = 1/sqrt(2)] */
  double b_interslice;    /* Relative neighbor weight along z direction [default = 1] */
  /* derived from basic parameters */
  double SigmaXsq;        /* derived parameter: SigmaX^2 */
  double pow_sigmaX_p;    /* pow(sigmaX,p) */
  double pow_sigmaX_q;    /* pow(sigmaX,q) */
  double pow_T_qmp;       /* pow(T,q-p) */

  /*--Specific to BM3D and CNN (Deep Res-net) --*/
  double QuantLevel_lower; /* Denoiser input must be normalized to (0,1). Lower and upper pixel bounds in image that map to 0 and 1 values */ 
  double QuantLevel_upper; 
  double Sigma_n;          /* on a scale of 0-->255 */

  /* others */
  char   DataDir[200];     /* Directory where denoiser-routine (Python script that is external to C executable) accesses input / output */ 

  /*--Specific to CNN --*/
  char TF_gpu_flag;             /* Run TensorFlow (TF) with GPU acceleration (default) */  
  char TF_CheckPointDir[256];   /* Directory where all checkpoint files (containing CNN model parameters) are stored */
  char TF_CheckpointState[256]; /* values are 'latest' or 'specific'. default='latest', loads most recent checkpoint */
  int  TF_CheckpointEpochNum;   /* Not needed if CheckpointState is 'latest'. Indicates epoch number for loading specific checkpoint. */ 


};

/* Reconstruction Parameters Data Structure */
struct ReconParams
{
  /* General parameters */
  double InitImageValue;  /* Initial Condition pixel value. In our examples usually chosen as ... */
  double StopThreshold;   /* Stopping threshold in percent */
  int MaxIterations;      /* Maximum number of iterations */
  int Positivity;         /* Positivity constraint: 1=yes, 0=no */
  
  /* sinogram weighting */
  double SigmaY;          /* Scaling constant for sinogram weights (e.g. W=exp(-y)/SigmaY^2 ) */
  int weightType;         /* How to compute weights if internal, 0: =1 (default); 1: exp(-y); 2: exp(-y/2) */

  char MBIRMode[50];     /* conventional or PnP */
  char PriorModel[50];   /* prior model: QGGMRF, BM3D or CNN. */

  struct PriorParams priorparams;    /* paramters for prior model */ 
  
  /* Relevent parameters if MBIRMode is "PnP" */
  double SigmaPnP;
  double RhoPnP;
  float **proximalmap; /* ptr to 3D proximal map image; here to carry it to the ICD update. initialized and maintained when reconstruction is initiated */
  /* derived */
  double SigmaPnPsq;
};


/* 2D Sinogram Data Structure */
struct Sino2DParallel
{
   struct SinoParams3DParallel sinoparams; /* Sinogram Parameters */
   float *sino;		/* Array of sinogram entries indexed by sino[NChannels*view + channel] */
   float *weight;	/* Weights for each measurement */
			/* If data arrays empty, then set the pointer = NULL */
};


/* 2D Image Data Structure */
struct Image2D
{
   struct ImageParams3D imgparams;	/* Image parameters */
   float *image;	/* Array of image entries indexed by image[Nx*row + column] */
			/* If data array is empty, then set image = NULL */
};

/* Sparse Column Vector - Data Structure */
struct SparseColumn
{
   int Nnonzero;	/* Nnonzero is the number of nonzero entries in the column */
   int *RowIndex;	/* RowIndex[j] is the row index of the jth nonzero entry in the column */
   float *Value;	/* Value[j] is the value of the jth nonzero entry in the column of the matrix */
};

/* Sparse System Matrix Data Structure */
struct SysMatrix2D
{
   int Ncolumns;		/* Number of columns in sparse matrix */
   struct SparseColumn *column;	/* column[i] is the i-th column of the matrix in sparse format */
};




#endif /* MBIR_MODULAR_DEFS_H*/

