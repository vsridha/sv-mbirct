/* ============================================================================== 
 * Copyright (c) 2018 Venkatesh Sridhar (Purdue University)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice, this
 * list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * Neither the name of Xiao Wang, Purdue University,
 * nor the names of its contributors may be used
 * to endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================== */

#include <math.h>
#include <omp.h>

#include "MBIRModularDefs.h"
#include "allocate.h"

#include "recon3d.h"
#include "initialize.h"
#include "icd3d.h"
#include "pnp_denoiser.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>



#define VERBOSE_FLAG_PRIOR_DENOISER 0

/* Internal functions */
void  one_way_shuffle(int *order, int len);
float ICDStep3D_Prior(struct PriorParams priorparams, float tempV, float *neighbors, float tempProxMap, float SigmaPnPsq);
void ExtractNeighbors_QGGMRF( float *neighbors, int jz, int jy, int jx, float **x, struct ImageParams3D imgparams);

/* Performs QGGMRF-denoising by proximal-updates */
void Proximal_QGGMRF_Denoising(struct Image3D *Image,             /* Initialization for proximal-update. At end of routine, stores denoiser output */
                               struct PriorParams priorparams,    /* prior parameters */ 
                               char   *ImageReconMask, 
                               struct Image3D *ProximalMapInput,  /* Input to denoiser */
                               float  SigmaPnP,                   /* analogous to denoiser "noise-level" parameter */       
                               int    NumICDPasses,               /* #ICD passes for evaluating proximal-update */  
                               char   positivity_flag)                    
{
    int it, j;
    int *order;
    float cost, avg_change;
    struct timeval start,stop;
    double tdiff;
    int Nyx=Image->imgparams.Nx * Image->imgparams.Ny; /* image size */
    int Nz =Image->imgparams.Nz;

    gettimeofday(&start,NULL);  /* starting time */
    
    /* Allocate memory for order of updates */
    order = (int *)get_spc(Nyx, sizeof(int));

    /* Order of pixel updates need NOT be raster order, just initialize */
    for (j = 0; j < Nyx; j++)
        order[j] = j;
        
    /****************************************/
    /* DENOISING using 1 or more ICD passes */
    /****************************************/
    if(VERBOSE_FLAG_PRIOR_DENOISER)
        printf("\nStarting QGGMRF Denoising  ... \n");

    for (it = 0; it < NumICDPasses; it++)
    {     
        one_way_shuffle(order, Nyx); 

        /* Update all voxels in ROI exactly once: Random update order in (jy,jx) provides the fastest algorithmic convergence */
        RandomizedICD_QGGMRF(Image, ProximalMapInput, ImageReconMask, priorparams, SigmaPnP, order, Nyx, Nz, positivity_flag, &avg_change);    

        if(VERBOSE_FLAG_PRIOR_DENOISER){
            cost = ProximalMapCostFunction3D_QGGMRF(Image, priorparams, ProximalMapInput, SigmaPnP);
            fprintf(stdout, "QGGMRF Denoiser proximal-iteration %-3d: cost = %-15f, Average change = %f %%\n", it+1, cost, avg_change);
        }
    }
   
    gettimeofday(&stop,NULL);  /* XW: stopping time */
    tdiff = 1000*(stop.tv_sec - start.tv_sec) + ((stop.tv_usec - start.tv_usec) / 1000);
    //fprintf(stdout, "Q-GGMRF denoising time = %-10f ms \n", tdiff);
    
    free((void *)order);    
}

/* The function to compute cost function of prior proximal map */
float ProximalMapCostFunction3D_QGGMRF(struct Image3D *Image, 
                                       struct PriorParams priorparams,
                                       struct Image3D *ProximalMapInput, 
                                       float  SigmaPnP)
{
    float SigmaPnPSq, ProximalCost=0.0;
    int i,jz,Nyx,Nz;
        
    float **image= Image->image;
    float **prox = ProximalMapInput->image;
    
    Nyx = Image->imgparams.Nx*Image->imgparams.Ny;
    Nz  = Image->imgparams.Nz;
    SigmaPnPSq = SigmaPnP * SigmaPnP ;

    for(jz=0;jz<Nz;jz++)
    for(i=0;i<Nyx;i++)
        ProximalCost += (image[jz][i]-prox[jz][i])*(image[jz][i]-prox[jz][i]);
    
    return (QGGMRFCostFunction3D(image, priorparams, Image->imgparams) + ProximalCost/(2*SigmaPnPSq)) ;
}   


void RandomizedICD_QGGMRF(struct Image3D *Image, 
                          struct Image3D *ProximalMapInput, 
                          char *ImageReconMask, 
                          struct PriorParams priorparams, 
                          float SigmaPnP, 
                          int *order, 
                          int Nyx, 
                          int Nz, 
                          char positivity_flag, 
                          float *avg_change)  
{
    float **x;       /* image data */
    float **xtilde;  /* proximal map input image */
    //struct ImageParams3D *imgparams;
    //float *neighbors;
    //float neighbors[10];
    int Nthreads;
    float **neighbors;

    int tid, jyx, jz, jx, jy, j,l, Nx;
    float pixel, diff,v;
    float SigmaPnPsq;
    float TotalValueChange = 0, TotalPixelValue=0;
    int NumUpdatedPixels=0;
    char zero_skip_FLAG;

    /* Assign local pointers to data */
    x=Image->image; 
    xtilde = ProximalMapInput->image;
    Nx=Image->imgparams.Nx;
    //neighbors=(float *)malloc(10*sizeof(float));
    SigmaPnPsq = SigmaPnP*SigmaPnP;


    #pragma omp parallel
    {
        Nthreads=omp_get_max_threads();
        /*
        #pragma omp single
            printf("Maximum Number of threads = %d\n",Nthreads);
        */
    }

    neighbors=(float **)malloc(Nthreads*sizeof(float *));
    for(j=0;j<Nthreads;j++)
        neighbors[j]=(float *)malloc(10*sizeof(float));

    /* One Full Pass of ICD  : update all pixels in PixelList */
    #pragma omp parallel
    { 
        #pragma omp for private(tid,jyx,jy,jx,jz,j,pixel,v,diff,zero_skip_FLAG) reduction(+:TotalValueChange,TotalPixelValue,NumUpdatedPixels)
        for (l = 0; l < Nyx; l++)
        {
            tid = omp_get_thread_num();
            jyx = order[l];
            if(ImageReconMask[jyx]) /* Pixel is within ROI */
            {
                jy = jyx/Nx;
                jx = jyx%Nx;

                for(jz=0;jz<Nz;jz++)
                {
                    v = x[jz][jyx];
                    ExtractNeighbors_QGGMRF(neighbors[tid], jz, jy, jx, x, Image->imgparams);

                    /* check for zero_skip */
                    zero_skip_FLAG=0;
                    if (v == 0.0)
                    {
                        zero_skip_FLAG=1;
                        for (j = 0; j < 10; j++)
                        {
                            if (neighbors[tid][j] != 0.0)
                            {
                                zero_skip_FLAG=0;
                                break; 
                            }
                        }
                    }
                    
                    if(zero_skip_FLAG == 0)
                    {
                        //printf("x=%f, xtilde=%f,neighbors[0]")
                        pixel = ICDStep3D_Prior(priorparams, x[jz][jyx], neighbors[tid], xtilde[jz][jyx], SigmaPnPsq);

                        if(positivity_flag && (pixel<0)) 
                           pixel=0;

                        x[jz][jyx] = pixel;       
                        diff = pixel - v;

                        TotalValueChange += fabs(diff);
                        TotalPixelValue += v ; 
                        NumUpdatedPixels += 1;
                    }
                }   
            }
        }
    }


    if(NumUpdatedPixels>0)
        *avg_change = (TotalValueChange/TotalPixelValue)*100;  

    for(j=0;j<Nthreads;j++)
        free(neighbors[j]);
                
} 

/* Single pixel update using ICD */
float ICDStep3D_Prior(struct PriorParams priorparams, float tempV, float *neighbors, float tempProxMap, float SigmaPnPsq)
{
    float theta1, theta2, UpdatedPixelValue;
    
    /* Formulate the quadratic surrogate function (with coefficients theta1, theta2) for the local cost function */
    theta1 = 0.0;
    theta2 = 0.0;
    
    QGGMRF3D_Update(priorparams, tempV, neighbors, &theta1, &theta2);
    PandP_Update(SigmaPnPsq, tempV, tempProxMap, &theta1, &theta2);

    /* Calculate Updated Pixel Value */
    UpdatedPixelValue = tempV - (theta1/theta2);
    
    return UpdatedPixelValue;
}

void ExtractNeighbors_QGGMRF( float *neighbors, int jz, int jy, int jx, float **x, struct ImageParams3D imgparams)
{
    int jyx=jy*imgparams.Nx+jx;
    int Nz=imgparams.Nz;

    ExtractNeighbors_WithinSlice(neighbors, jx, jy, &x[jz][0], imgparams);

    neighbors[8]  = (jz>0) ? x[jz-1][jyx] : x[Nz-1][jyx]  ;
    neighbors[9] = (jz<(Nz-1)) ? x[jz+1][jyx] : x[0][jyx] ;  
}


void one_way_shuffle(int *order, int len)
{
    int i, j, tmp;

    for (i = 0; i < len-1; i++)
    {
        j = i + (rand() % (len-i));
        tmp = order[j];
        order[j] = order[i];
        order[i] = tmp;
    }
}


/* Non-optimization Denoiser : BM3D */
void BM3DDenoise(struct Image3D *CleanImage, struct Image3D *NoisyImage, struct PriorParams priorparams) /* Input - noisy image. Output - clean (denoised) image */
{
    char SysCommand[200];
    char SrcDir[200];
    char DataDir[200];
    char fname[200];

    float lower = (float)priorparams.QuantLevel_lower;
    float upper = (float)priorparams.QuantLevel_upper;
    float Sigma_n = (float)priorparams.Sigma_n;

    struct ImageParams3D *imgparams = &(NoisyImage->imgparams);

    /* for now set these file paths (wrt run-script) locally instead of getting them from parameters */
    strcpy(SrcDir, "../src");
    strcpy(DataDir, strcat(priorparams.DataDir,"_temp"));

    if(VERBOSE_FLAG_PRIOR_DENOISER)
        printf("\nBM3D Denoising \n");

    /* You can lkater get rid of having separate temp_data_in and temp_data_out files, and just have 1 set of temp_data files */
    /* Write noisy image */
    sprintf(fname, "%s_in", DataDir);
    if(WriteImage3D(fname, NoisyImage))
    {
         fprintf(stderr, "Error in writing out denoiser input image file through function BM3DDenoise \n");
         exit(-1);
    }

    /* Execute BM3D denoising through a Python script (located in src directory) */
    sprintf(SysCommand, "python3 %s/bm3d_wrapper.py -f %s -l %f -u %f -s %f -z %d -y %d -x %d -i %d -d %d", SrcDir, DataDir, lower, upper, Sigma_n, \
        imgparams->Nz, imgparams->Ny, imgparams->Nx, imgparams->FirstSliceNumber, imgparams->NumSliceDigits);
    system(SysCommand);

    /* Read in clean image */
    sprintf(fname, "%s_out", DataDir);

    if(ReadImage3D(fname, CleanImage))
    {
         fprintf(stderr, "Error in reading in clean image file through function BM3DDenoise \n");
         exit(-1);
    }

    /* Remove existing temp data (shift this to the end) */
    sprintf(SysCommand, "rm -r %s*.2Dimgdata", DataDir);
    system(SysCommand);
}


