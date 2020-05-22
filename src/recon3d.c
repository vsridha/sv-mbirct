
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>

#include "mbir_ct.h"
#include "MBIRModularDefs.h"
#include "MBIRModularUtils.h"
#include "allocate.h"
#include "icd3d.h"
#include "heap.h"
#include "A_comp.h"
#include "initialize.h"
#include "recon3d.h"
#include "pnp_denoiser.h"

/* Parameters for NH-updates */
#define  c_ratio 0.07
#define convergence_rho 0.7

/* Internal functions */

/* ICD Update for a single Super-voxel */
void super_voxel_recon(int jj,struct SVParams svpar,unsigned long *NumUpdates,float *totalValue,float *totalChange,int NH_flag,int *phaseMap,
	int *order,int *indexList,float **w,float **e,
	struct AValues_char ** A_Padded_Map,float *max_num_pointer,struct heap_node *headNodeArray,
	struct SinoParams3DParallel sinoparams,struct ReconParams reconparams,struct Image3D *Image,
	float *voxelsBuffer1,float *voxelsBuffer2,int* group_array,int group_id);

/* ICD update exactly once for all Super-voxels (1 effective ICD iteration or 1 equit) */
void MAP_or_Likelihood_Inversion_ICD_SingleEquit(float **e, float **w, struct SinoParams3DParallel sinoparams, struct Image3D *Image, \
	struct ReconParams reconparams, struct AValues_char **A_Padded_Map, float *max_num_pointer, struct SVParams svpar, int mix_of_NH_and_H_updates_flag, \
	int *phaseMap, int group_id_list[][4], int *order, int *indexList, struct heap_node *headNodeArray, int rep_num, int repnum_factor, \
	int indexList_size, float *voxelsBuffer1, float *voxelsBuffer2, unsigned long *NumUpdates_ext, float *totalValue_ext, float *totalChange_ext);

void coordinateShuffle(int *order1, int *order2,int len);
void three_way_shuffle(int *order1, int *order2,struct heap_node *headNodeArray,int len);
float MAPCostFunction3D(float **e,struct Image3D *Image,struct Sino3DParallel *sinogram,struct ReconParams *reconparams);
void MannUpdate(float **img_current, float **img_previous, float Rho, struct ImageParams3D *imgparams);
void Reflect(float **img_reflected, float **img_in, float **img_ref, struct ImageParams3D *imgparams);


void MBIRReconstruct3D(
	struct Image3D *Image,
	struct Sino3DParallel *sinogram,
	float **e,  /* e=y-Ax, error */
	struct ReconParams reconparams,
	struct SVParams svpar,
	struct AValues_char ** A_Padded_Map,
	float *max_num_pointer,
	char *ImageReconMask,
	struct CmdLine *cmdline)
{
	int i,j,jj,p,t,it,it_print=1;
	int NumMaskVoxels=0;
	float **x;  /* image data */
	float **y;  /* sinogram projections data */
	float **w;  /* projections weights data */
	float *voxelsBuffer1;  /* the first N entries are the voxel values.  */
	float *voxelsBuffer2;
	unsigned long NumUpdates=0;
	float totalValue=0,totalChange=0,equits=0;
	float avg_update,avg_update_rel;

	int *order;
	struct timeval tm1,tm2;

	x = Image->image;     /* x is the image vector */
	y = sinogram->sino;   /* y is the sinogram vector */
	w = sinogram->weight; /* vector of weights for each sinogram measurement */
	int Nx = Image->imgparams.Nx;
	int Ny = Image->imgparams.Ny;
	int Nxy = Nx*Ny;
	int Nz = Image->imgparams.Nz;
	int NvNc = sinogram->sinoparams.NViews * sinogram->sinoparams.NChannels;
	int NViews = sinogram->sinoparams.NViews;
	int MaxIterations = reconparams.MaxIterations;
	float StopThreshold = reconparams.StopThreshold;
	int SVLength = svpar.SVLength;
	int overlappingDistance = svpar.overlap;
	int SV_depth = svpar.SVDepth;
	int SV_per_Z = svpar.SV_per_Z;
	int SVsPerLine = svpar.SVsPerLine;
	int sum = svpar.Nsv;
	int pieceLength = svpar.pieceLength;
	struct minStruct * bandMinMap = svpar.bandMinMap; /* for each SV, sinogram-trace that lower-bounds that of entire SV */
	struct maxStruct * bandMaxMap = svpar.bandMaxMap; /* for each SV, sinogram-trace that upper-bounds that of entire SV */
	
	/* Added */
	int MBIRMode;
	int PriorModel;
	int jz,jxy,subit=0,rep_num_factor=1,group=0,positivity_flag;
	float RhoPnP,residue;
	struct Image3D W, W_prev, V, Z;
	struct Image3D *X; 
	struct ImageParams3D *imgparams=&(Image->imgparams);

	if(strcmp(reconparams.MBIRMode, "conventional")==0) 
		MBIRMode = MBIR_MAP_ESTIMATION; 
	else if(strcmp(reconparams.MBIRMode, "PnP")==0) 
		MBIRMode = MBIR_PnP_PRIORS;
	else
	{
		fprintf(stderr, "Error: Unrecognized MBIR mode %s \n", reconparams.MBIRMode);
		exit(-1);
	}

	if(MBIRMode==MBIR_PnP_PRIORS)
	{
		/* Assign pointer*/
		X=Image;

		/* Allocate memory */
		W.imgparams.Nx = Nx;
		W.imgparams.Ny = Ny;
		W.imgparams.Nz = Nz;
		W.imgparams.FirstSliceNumber = imgparams->FirstSliceNumber;
		W.imgparams.NumSliceDigits   = imgparams->NumSliceDigits;
		AllocateImageData3D(&W);

		W_prev.imgparams.Nx = Nx;
		W_prev.imgparams.Ny = Ny;
		W_prev.imgparams.Nz = Nz;
		W_prev.imgparams.FirstSliceNumber = imgparams->FirstSliceNumber;
		W_prev.imgparams.NumSliceDigits   = imgparams->NumSliceDigits;
		AllocateImageData3D(&W_prev);

		V.imgparams.Nx = Nx;
		V.imgparams.Ny = Ny;
		V.imgparams.Nz = Nz;
		V.imgparams.FirstSliceNumber = imgparams->FirstSliceNumber;
		V.imgparams.NumSliceDigits   = imgparams->NumSliceDigits;
		AllocateImageData3D(&V);

		Z.imgparams.Nx = Nx;
		Z.imgparams.Ny = Ny;
		Z.imgparams.Nz = Nz;
		Z.imgparams.FirstSliceNumber = imgparams->FirstSliceNumber;
		Z.imgparams.NumSliceDigits   = imgparams->NumSliceDigits;
		AllocateImageData3D(&Z);

		/* Initialize */
		for(jz=0;jz<Nz;jz++)
		{
			memcpy(W.image[jz],Image->image[jz],Nxy*sizeof(float));
			memcpy(W_prev.image[jz],Image->image[jz],Nxy*sizeof(float));
			memcpy(V.image[jz],Image->image[jz],Nxy*sizeof(float));
			/* Z initialized later */
		}

		if(strcmp(reconparams.PriorModel, "QGGMRF")==0)
			PriorModel = PRIOR_TYPE_QGGMRF;
		else if(strcmp(reconparams.PriorModel, "BM3D")==0)
			PriorModel = PRIOR_TYPE_BM3D;
		else if(strcmp(reconparams.PriorModel, "CNN")==0)
			PriorModel = PRIOR_TYPE_CNN;
		else
		{
			fprintf(stderr, "Error: PriorModel %s not recognized \n", reconparams.PriorModel);
			exit(-1);
		}
	}

	int rep_num=(int)ceil(1/(4*c_ratio*convergence_rho));  /* VS: this is 10 for the parameters defined above */		

    for(j=0;j<Nxy;j++)
    if(ImageReconMask[j])
            NumMaskVoxels++;

	/* Order of pixel updates need NOT be raster order, just initialize */
	order = (int *)_mm_malloc(sum*SV_per_Z*sizeof(int),64);

	t=0;

	/* To generate random order of SV-updates */
	for(p=0;p<Nz;p+=SV_depth)
	for(i=0;i<Ny;i+=(SVLength*2-overlappingDistance))
	for(j=0;j<Nx;j+=(SVLength*2-overlappingDistance))
	{
		order[t]=p*Nxy+i*Nx+j;  /* order is the first voxel coordinate, not the center */
		t++;
	}

	int phaseMap[sum*SV_per_Z];

	/* Tiled pattern of SV-updates : SVs of a particular tile color can be updated simultaneosuly */
	#pragma omp parallel for private(jj) schedule(dynamic)
	for(i=0;i<SV_per_Z;i++)
	for(jj=0;jj<sum;jj++)
	{
		if((jj/SVsPerLine)%2==0)
		{
			if((jj%SVsPerLine)%2==0)
				phaseMap[i*sum+jj]=0;
			else
				phaseMap[i*sum+jj]=1;			
		}
		else
		{
			if((jj%SVsPerLine)%2==0)
				phaseMap[i*sum+jj]=2;
			else
				phaseMap[i*sum+jj]=3;			
		}
	}

	int group_id_list[SV_per_Z][4];

	/* Sequence in which the 4 tile colors are specified for SV-updates */
	/* Reconstruction can be split into super-slices, each of consisting of "sv-depth" consecutive slices */
	/* For each super-slice, ONLY SVs of same tile-color can be updated at a time */
	/* However, at a time the chosen tile-color across super-slices can vary */
	for(i=0;i<SV_per_Z;i++){
		if(i%4==0){
			group_id_list[i][0]=0;
			group_id_list[i][1]=3;
			group_id_list[i][2]=1;
			group_id_list[i][3]=2;										
		}
		else if(i%4==1){	
			group_id_list[i][0]=3;
			group_id_list[i][1]=0;
			group_id_list[i][2]=2;
			group_id_list[i][3]=1;										
		}
		else if(i%4==2){			
			group_id_list[i][0]=1;
			group_id_list[i][1]=2;
			group_id_list[i][2]=3;
			group_id_list[i][3]=0;										
		}
		else{
			group_id_list[i][0]=2;
			group_id_list[i][1]=1;
			group_id_list[i][2]=0;
			group_id_list[i][3]=3;										
		}				
	}
	srand(time(NULL));
	/* Heap used for non-homogenous SV updates */
	struct heap_node headNodeArray[sum*SV_per_Z];

	for(i=0;i<SV_per_Z;i++)
	for(jj=0;jj<sum;jj++)
	{
		headNodeArray[i*sum+jj].pt=i*sum+jj;
		headNodeArray[i*sum+jj].x=0.0;
	}
	int indexList_size=(int) sum*SV_per_Z*4*c_ratio*(1-convergence_rho);	/* ~20% of #voxels for the parameters defined above */

	indexList_size/=2;														/* ~10% of #voxels for the parameters define above */
	rep_num_factor=2;
	
	int indexList[indexList_size];   	             	    
    

	voxelsBuffer1 = (float *)_mm_malloc(Nxy*sizeof(float),64);
	voxelsBuffer2 = (float *)_mm_malloc(Nxy*sizeof(float),64);

	for(i=0;i<Nxy;i++) voxelsBuffer1[i]=0;
	for(i=0;i<Nxy;i++) voxelsBuffer2[i]=0;

	it=0;

	coordinateShuffle(&order[0],&phaseMap[0],sum*SV_per_Z);
	
	int startIndex=0;
	int endIndex=0;        		
         
	char stop_FLAG=0;

	/* BEGIN: main MBIR loop */
	//printf("Started main reconstruction \n");
	while(stop_FLAG==0 && equits<MaxIterations && it<MaxIterations)
	{	

		if(MBIRMode==MBIR_PnP_PRIORS)
		{
			/* porximal-input for CT likelihood-inversion agent F */
			reconparams.proximalmap = W.image;
			/* make a copy of W: to be used later for Mann update */
			for(jz=0;jz<Nz;jz++)
				memcpy(W_prev.image[jz],W.image[jz],Nxy*sizeof(float));
		}

		/* ---In case of reconparams.MBIRMode==MBIR_PnP_PRIORS, the below snippet gives us F(W), where F is CT likelihood-inversion agent and W is proximal input ---*/
		/* ---In case of reconparams.MBIRMode==MBIR_MAP_ESTIMATION, the below snippet gives us the iterative update for the MAP estimate --*/	
		MAP_or_Likelihood_Inversion_ICD_SingleEquit(e, w, sinogram->sinoparams, Image, reconparams, A_Padded_Map, max_num_pointer, svpar, \
		 (int)(it>0), &phaseMap[0], group_id_list, order, &indexList[0], &headNodeArray[0], rep_num, rep_num_factor, indexList_size, \
		 voxelsBuffer1, voxelsBuffer2, &NumUpdates, &totalValue, &totalChange);


		if(MBIRMode==MBIR_PnP_PRIORS)
		{

			RhoPnP  = (it<1)?0.5:reconparams.RhoPnP;
			positivity_flag = (it<1)?0:reconparams.Positivity;
			/* Initialize Z (needed only for proximal-denoisers) */
			if(it==0)
			{	
				for(jz=0;jz<Nz;jz++)
					memcpy(Z.image[jz],X->image[jz],Nxy*sizeof(float));
			}

			/* V=2X-W, where X=F(W). So below step gives us V=(2F-I)W */
			Reflect(V.image, X->image, W.image, imgparams);
			/* Z=H(V)*/
			if(PriorModel==PRIOR_TYPE_QGGMRF)
				Proximal_QGGMRF_Denoising(&Z, reconparams.priorparams, ImageReconMask, &V, reconparams.SigmaPnP, 1, positivity_flag);
			else if(PriorModel==PRIOR_TYPE_BM3D)
				BM3DDenoise(&Z, &V, reconparams.priorparams);
			else if(PriorModel==PRIOR_TYPE_CNN)
				CNNDenoise(&Z, &V, reconparams.priorparams);

			/* W=2Z-V, where Z=H(V) and V=(2F-I)W. So this step gives W=(2H-I)(2F-I)W. */
			Reflect(W.image, Z.image, V.image, imgparams);
			/* Mann update on W */
			MannUpdate(W.image, W_prev.image, RhoPnP, imgparams);
		}
		
		if(NumUpdates>0)
		{
			avg_update = totalChange/NumUpdates;
			float avg_value = totalValue/NumUpdates;
			avg_update_rel = avg_update/avg_value * 100;

			if(MBIRMode==MBIR_PnP_PRIORS)
			{
				residue=0;
				for(jz=0;jz<Nz;jz++)
				for(jxy=0;jxy<Nxy;jxy++)
				{
					if(ImageReconMask[jxy])
					residue += (X->image[jz][jxy]-Z.image[jz][jxy])*(X->image[jz][jxy]-Z.image[jz][jxy]);
				}
				residue = sqrt(residue/(Nz*NumMaskVoxels)); 
			}

		}
		
		if (avg_update_rel < StopThreshold && (endIndex!=0))
			stop_FLAG = 1;

		// Increment iteration (each iterayion < 2 equit if MBIR_MAP_ESTIMATION and < 1 equit if MBIR_PnP_Priors)
		it++;
		equits += (float)NumUpdates/(NumMaskVoxels*Nz);
		if(equits > it_print)
		{
			if(MBIRMode==MBIR_PnP_PRIORS)
				fprintf(stdout,"\titeration %d, average change %.4f %%, PnP residue %.4f mm^-1\n",it_print,avg_update_rel,residue);
			else
				fprintf(stdout,"\titeration %d, average change %.4f %%\n",it_print,avg_update_rel);

			it_print++;
		}

		NumUpdates=0;
		totalValue=0;
		totalChange=0;

	}/* END: main MBIR loop */
	
	if(StopThreshold <= 0)
		fprintf(stdout,"\tNo stopping condition--running fixed iterations\n");
	else if(stop_FLAG == 1)
		fprintf(stdout,"\tReached stopping condition\n");
	else
		fprintf(stdout,"\tWARNING: Didn't reach stopping condition\n");

	fprintf(stdout,"\tEquivalent iterations = %.1f, (non-homogeneous iterations = %d)\n",equits,it);
	fprintf(stdout,"\tAverage update in last iteration (relative) = %f %%\n",avg_update_rel);
	fprintf(stdout,"\tAverage update in last iteration (magnitude) = %f mm^-1\n",avg_update);
	
	_mm_free((void *)order);
	_mm_free((void *)voxelsBuffer1);
	_mm_free((void *)voxelsBuffer2);

}   /*  END MBIRReconstruct3D()  */


/* ---In case reconparams.MBIRMode==MBIR_PnP_PRIORS, 	 we perform 1 equit (effective ICD iteration) of Likelihood inversion (via proximal-map computation) ---*/
/* ---In case reconparams.MBIRMode==MBIR_MAP_ESTIMATION, we perform 1 equit of MAP inversion using QGGMRF as prior model ---*/
void MAP_or_Likelihood_Inversion_ICD_SingleEquit(
	float **e,
	float **w,
	struct SinoParams3DParallel sinoparams,
	struct Image3D *Image,
	struct ReconParams reconparams,
	struct AValues_char **A_Padded_Map,
	float *max_num_pointer,
	struct SVParams svpar,
	int mix_of_NH_and_H_updates_flag,
	int *phaseMap,
	int group_id_list[][4],
	int *order,
	int *indexList,
	struct heap_node *headNodeArray,
	int rep_num,
	int rep_num_factor,
	int indexList_size,
	float *voxelsBuffer1, // voxelsBuffer1,2 not needed for single-node reconstruction
	float *voxelsBuffer2,
	unsigned long *NumUpdates_ext,
	float *totalValue_ext,
	float *totalChange_ext
	)
{
	int i,jj,group,startIndex,endIndex,subit;
	unsigned long NumUpdates=0;
	float totalValue=0, totalChange=0;
	int SV_per_Z = svpar.SV_per_Z;
	int sum = svpar.Nsv;
	int NH_flag;

	struct heap priorityheap;
	initialize_heap(&priorityheap);

	/* Do only Homogenous SV updates */
	if(!mix_of_NH_and_H_updates_flag)
	{
		NH_flag=0;
		startIndex=0;
		endIndex=sum*SV_per_Z;

		/* Which SVs in the above list can be updated simulatenously ? */
		/* We cover all SVs by iterating through 4 cycles */
		/* A given SV with (super)slice-index jz is updated if its tile-color (called "phase-map" here) matches that specified by group_id_list(jz,cycle)  */
		for (group = 0; group < 4; group++)
		{
			#pragma omp parallel for schedule(dynamic)  reduction(+:NumUpdates) reduction(+:totalValue) reduction(+:totalChange)
			for (jj = startIndex; jj < endIndex; jj+=1)
				super_voxel_recon(jj,svpar,&NumUpdates,&totalValue,&totalChange,NH_flag,&phaseMap[0],order,&indexList[0],w,e,A_Padded_Map,&max_num_pointer[0],&headNodeArray[0],sinoparams,reconparams,Image,voxelsBuffer1,voxelsBuffer2,&group_id_list[0][0],group);
		}
	}	
	else /* Do a mix of Homogenous and Non-homogenous SV updates */
	{
		subit=1;

		/* Each iteration divided into 10 cycles (sub-iterations) */
		while(subit<=10)
		{
			/* after every 10th cycle (sub-iteration), perform below shuffle */
			if(subit==1)	/* Removed (it!=1) condition */
				three_way_shuffle(&order[0],&phaseMap[0],&headNodeArray[0],sum*SV_per_Z);

			/* Alternating cycles of homogenous and Non-homogenous updates (10 alternating cycles, in each cycle 20% of SVs updated  */
			if(subit%2==1)
			{
				/* Non-homogenous updates */
				NH_flag=1;
				initialize_heap(&priorityheap);						
				for(jj=0;jj<sum*SV_per_Z;jj++){
					heap_insert(&priorityheap, &(headNodeArray[jj]));
				}						
				startIndex=0;					
				endIndex=indexList_size;					

				for(i=0;i<endIndex;i++){
					struct heap_node tempNode;
					get_heap_max(&priorityheap, &tempNode);
					indexList[i]=tempNode.pt;
				}	
			}				
			else{	
				/* Homogenous updates (random order) */
				NH_flag=0;				
				startIndex=((subit-2)/2)%(rep_num*rep_num_factor)*sum*SV_per_Z/(rep_num*rep_num_factor);
				endIndex=(((subit-2)/2)%(rep_num*rep_num_factor)+1)*sum*SV_per_Z/(rep_num*rep_num_factor);
			}

			/* Which SVs in the above list can be updated simulatenously ? */
			/* We cover all SVs by iterating through 4 cycles */
			/* A given SV with (super)slice-index jz is updated if its tile-color (called "phase-map" here) matches that specified by group_id_list(jz,cycle)  */
			for (group = 0; group < 4; group++)
			{
				#pragma omp parallel for schedule(dynamic)  reduction(+:NumUpdates) reduction(+:totalValue) reduction(+:totalChange)
				for (jj = startIndex; jj < endIndex; jj+=1)
					super_voxel_recon(jj,svpar,&NumUpdates,&totalValue,&totalChange,NH_flag,&phaseMap[0],order,&indexList[0],w,e,A_Padded_Map,&max_num_pointer[0],&headNodeArray[0],sinoparams,reconparams,Image,voxelsBuffer1,voxelsBuffer2,&group_id_list[0][0],group);
			}

			subit++;
		}
	}
	*NumUpdates_ext = NumUpdates;
	*totalChange_ext = totalChange;
	*totalValue_ext = totalValue;

	if(priorityheap.size>0)
		free_heap((void *)&priorityheap); 
}
			

void forwardProject2D(
	float *e,
	float *x,
	struct AValues_char ** A_Padded_Map,
	float *max_num_pointer,
	struct SinoParams3DParallel *sinoparams,
	struct ImageParams3D *imgparams,
	struct SVParams svpar)
{
	int jx,jy,Nx,Ny,i,M,r,j,p,SVNumPerRow;
	float inverseNumber=1.0/255;
	int SVLength = svpar.SVLength;
	int overlappingDistance = svpar.overlap;
	struct minStruct * bandMinMap = svpar.bandMinMap;
	int pieceLength = svpar.pieceLength;

	const int NViewsdivided=(sinoparams->NViews)/pieceLength;

	Nx = imgparams->Nx;
	Ny = imgparams->Ny;
	M = sinoparams->NViews*sinoparams->NChannels;

	for (i = 0; i < M; i++)
		e[i] = 0.0;

	if((Nx%(2*SVLength-overlappingDistance))==0)
		SVNumPerRow=Nx/(2*SVLength-overlappingDistance);
	else
		SVNumPerRow=Nx/(2*SVLength-overlappingDistance)+1;

	for (jy = 0; jy < Ny; jy++)
	for (jx = 0; jx < Nx; jx++)
	{
		int temp1=jy/(2*SVLength-overlappingDistance);
		if(temp1==SVNumPerRow)  // I don't think this will happen
			temp1=SVNumPerRow-1;

		int temp2=jx/(2*SVLength-overlappingDistance);
		if(temp2==SVNumPerRow)  // I don't think this will happen
			temp2=SVNumPerRow-1;

		int SVPosition=temp1*SVNumPerRow+temp2;
 
		int SV_jy=temp1*(2*SVLength-overlappingDistance);
		int SV_jx=temp2*(2*SVLength-overlappingDistance);
		int VoxelPosition=(jy-SV_jy)*(2*SVLength+1)+(jx-SV_jx);
		/*
		fprintf(stdout,"jy %d jx %d SVPosition %d SV_jy %d SV_jx %d VoxelPosition %d \n",jy,jx,SVPosition,SV_jy,SV_jx,VoxelPosition);
		*/
		// I think the second condition will always be true
		if (A_Padded_Map[SVPosition][VoxelPosition].length > 0 && VoxelPosition < ((2*SVLength+1)*(2*SVLength+1)))
		{
			/*XW: remove the index field in struct ACol and exploit the spatial locality */
			unsigned char* A_padd_Tranpose_pointer = &A_Padded_Map[SVPosition][VoxelPosition].val[0];
			for(p=0;p<NViewsdivided;p++) 
			{
				const int myCount=A_Padded_Map[SVPosition][VoxelPosition].pieceWiseWidth[p];
				int position=p*pieceLength*sinoparams->NChannels+A_Padded_Map[SVPosition][VoxelPosition].pieceWiseMin[p];

				for(r=0;r<myCount;r++)
				for(j=0;j< pieceLength;j++)
				{
					if((A_Padded_Map[SVPosition][VoxelPosition].pieceWiseMin[p]+bandMinMap[SVPosition].bandMin[p*pieceLength+j]+r)>=sinoparams->NChannels)
						fprintf(stdout, "p %d r %d j %d total_1 %d \n",p,r,j,A_Padded_Map[SVPosition][VoxelPosition].pieceWiseMin[p]+bandMinMap[SVPosition].bandMin[p*pieceLength+j]+r);

					if((position+j*sinoparams->NChannels+bandMinMap[SVPosition].bandMin[p*pieceLength+j]+r)>= M)
						fprintf(stdout, "p %d r %d j %d total_2 %d \n",p,r,j,position+j*sinoparams->NChannels+bandMinMap[SVPosition].bandMin[p*pieceLength+j]+r);

					if((position+j*sinoparams->NChannels+bandMinMap[SVPosition].bandMin[p*pieceLength+j]+r)< M)
						e[position+j*sinoparams->NChannels+bandMinMap[SVPosition].bandMin[p*pieceLength+j]+r] += A_padd_Tranpose_pointer[r*pieceLength+j]*max_num_pointer[jy*Nx+jx]*inverseNumber*x[jy*Nx+jx];

				}
				A_padd_Tranpose_pointer+=myCount*pieceLength;
			}
		}
	}

}   /* END forwardProject2D() */


void super_voxel_recon(
	int jj,
	struct SVParams svpar,
	unsigned long *NumUpdates,
	float *totalValue,
	float *totalChange,
	int NH_flag,
	int *phaseMap,
	int *order,
	int *indexList,
	float **w,
	float **e,
	struct AValues_char ** A_Padded_Map,
	float *max_num_pointer,
	struct heap_node *headNodeArray,
	struct SinoParams3DParallel sinoparams,
	struct ReconParams reconparams,
	struct Image3D *Image,
	float *voxelsBuffer1,
	float *voxelsBuffer2,
	int *group_array,
	int group_id)
{

	int jy,jx,p,i,q,t,j,currentSlice,startSlice;
	int SV_depth_modified;
	int NumUpdates_loc=0;
	float totalValue_loc=0,totalChange_loc=0;

	float ** image = Image->image;
	float ** proximalmap = reconparams.proximalmap;
	struct ImageParams3D imgparams = Image->imgparams;
	int Nx = imgparams.Nx;
	int Ny = imgparams.Ny;
	int Nz = imgparams.Nz;
	char PositivityFlag = reconparams.Positivity;

	int SVLength = svpar.SVLength;
	int overlappingDistance = svpar.overlap;
	int SV_depth = svpar.SVDepth;
	int SVsPerLine = svpar.SVsPerLine;
	struct minStruct * bandMinMap = svpar.bandMinMap;
	struct maxStruct * bandMaxMap = svpar.bandMaxMap;
	int pieceLength = svpar.pieceLength;
	int NViewsdivided = sinoparams.NViews/pieceLength;
	int MBIRMode;

	if(strcmp(reconparams.MBIRMode, "conventional")==0) 
		MBIRMode = MBIR_MAP_ESTIMATION; 
	else if(strcmp(reconparams.MBIRMode, "PnP")==0) 
		MBIRMode = MBIR_PnP_PRIORS;
	else
	{
		fprintf(stderr, "Error: Unrecognized MBIR mode %s \n", reconparams.MBIRMode);
		exit(-1);
	}

	if(!NH_flag)
	{	/* Homogenous update */
		startSlice = order[jj] / Nx / Ny; /* order[jj] - coordinate of first voxel within SV  */
		jy = (order[jj] - startSlice* Nx * Ny) / Nx;  
		jx = (order[jj] - startSlice* Nx * Ny) % Nx;
	}
	else
	{	/* Non-homogenous update */
		startSlice = order[indexList[jj]] / Nx / Ny;
		jy=(order[indexList[jj]] - startSlice* Nx * Ny) /Nx;
		jx=(order[indexList[jj]] - startSlice* Nx * Ny) %Nx;	
	}

	if((startSlice+SV_depth)>Nz)
		SV_depth_modified=Nz-startSlice;
	else
		SV_depth_modified=SV_depth;

	int theSVPosition=jy/(2*SVLength-overlappingDistance)*SVsPerLine+jx/(2*SVLength-overlappingDistance);
	if(!NH_flag)
	{
		if(phaseMap[jj]!=group_array[startSlice/SV_depth*4+group_id])
			return;
	}
	else
	{
		if(phaseMap[indexList[jj]]!=group_array[startSlice/SV_depth*4+group_id])
			return;
	}

	int countNumber=0;		/* no. of of voxels to update within SV  */
	int radius =SVLength;	/* half-length of square SV */
	int coordinateSize=1;	
	if(radius!=0)
		coordinateSize=(2*radius+1)*(2*radius+1);
	int k_newCoordinate[coordinateSize];
	int j_newCoordinate[coordinateSize];
	int j_newAA=0;
	int k_newAA=0;
	int voxelIncrement=0;

	/* choosing the voxels within SV to be updated */
	for(j_newAA=jy;j_newAA<=(jy+2*radius);j_newAA++)
	for(k_newAA=jx;k_newAA<=(jx+2*radius);k_newAA++)
	{
		if(j_newAA>=0 && k_newAA >=0 && j_newAA <Ny && k_newAA < Nx)
		{
			if(A_Padded_Map[theSVPosition][voxelIncrement].length >0) {
				j_newCoordinate[countNumber]=j_newAA;
				k_newCoordinate[countNumber]=k_newAA;
				countNumber++;
			} 
		}
		voxelIncrement++;
	}

	/*Skip upating SV if no valid voxels are found */
	if(countNumber==0)
		return;

	coordinateShuffle(&j_newCoordinate[0],&k_newCoordinate[0],countNumber);

	/*XW: for a supervoxel, bandMin records the starting position of the sinogram band at each view*/
	/*XW: for a supervoxel, bandMax records the end position of the sinogram band at each view */
	int bandMin[sinoparams.NViews]__attribute__((aligned(32)));
	int bandMax[sinoparams.NViews]__attribute__((aligned(32)));
	int bandWidthTemp[sinoparams.NViews]__attribute__((aligned(32)));
	int bandWidth[NViewsdivided]__attribute__((aligned(32)));

	#ifdef USE_INTEL_MEMCPY
	_intel_fast_memcpy(&bandMin[0],&bandMinMap[theSVPosition].bandMin[0],sizeof(int)*(sinoparams.NViews));
	_intel_fast_memcpy(&bandMax[0],&bandMaxMap[theSVPosition].bandMax[0],sizeof(int)*(sinoparams.NViews)); 
	#else
	memcpy(&bandMin[0],&bandMinMap[theSVPosition].bandMin[0],sizeof(int)*(sinoparams.NViews));
	memcpy(&bandMax[0],&bandMaxMap[theSVPosition].bandMax[0],sizeof(int)*(sinoparams.NViews));
	#endif

	#pragma vector aligned 
	for(p=0;p< sinoparams.NViews;p++)
		bandWidthTemp[p]=bandMax[p]-bandMin[p];

	/* Sinogram buffer is split into multiple blocks, each having "piceLength" no. of views */
	/* Calculate the block-wise height of the sinogram buffer */
	for (p = 0; p < NViewsdivided; p++)
	{
		int bandWidthMax=bandWidthTemp[p*pieceLength];
		for(t=0;t<pieceLength;t++){
			if(bandWidthTemp[p*pieceLength+t]>bandWidthMax)
				bandWidthMax=bandWidthTemp[p*pieceLength+t];
		}
		bandWidth[p]=bandWidthMax;
	}

	int tempCount=0;

	/* Total size of sinogram buffer across all blocks */
	#pragma vector aligned
	#pragma simd reduction(+:tempCount) 
	for (p = 0; p < NViewsdivided; p++)
		tempCount+=bandWidth[p]*pieceLength;

	float ** newWArray = (float **)malloc(sizeof(float *) * NViewsdivided);
	float ** newEArray = (float **)malloc(sizeof(float *) * NViewsdivided);
	float ** CopyNewEArray = (float **)malloc(sizeof(float *) * NViewsdivided); 

	/* Block-wise memory allocation for sinogram buffer (inlcudes weights w, residual-sinogram e and updated version of e) */
	for (p = 0; p < NViewsdivided; p++)
	{
		newWArray[p] = (float *)malloc(sizeof(float)*bandWidth[p]*pieceLength*SV_depth_modified);
		newEArray[p] = (float *)malloc(sizeof(float)*bandWidth[p]*pieceLength*SV_depth_modified);
		CopyNewEArray[p] = (float *)malloc(sizeof(float)*bandWidth[p]*pieceLength*SV_depth_modified);
	}

	float *newWArrayPointer=&newWArray[0][0];
	float *newEArrayPointer=&newEArray[0][0];

	const int n_theta=sinoparams.NViews;

	/* From the main sinogram copy data relevant to this specific SV-update into buffer  */
	/* For view indexed by (p,t) where p=block-index and t=view-index within block, relevant channels are specified by bandMin(p,t)+bandWidth(p) */
	for (p = 0; p < NViewsdivided; p++)
	{
		newWArrayPointer=&newWArray[p][0];
		newEArrayPointer=&newEArray[p][0];
		for(i=0;i<SV_depth_modified;i++)
		for(q=0;q<pieceLength;q++) 
		{
			#ifdef USE_INTEL_MEMCPY
			_intel_fast_memcpy(newWArrayPointer,&w[startSlice+i][p*pieceLength*sinoparams.NChannels+q*sinoparams.NChannels+bandMin[p*pieceLength+q]],sizeof(float)*(bandWidth[p]));
			_intel_fast_memcpy(newEArrayPointer,&e[startSlice+i][p*pieceLength*sinoparams.NChannels+q*sinoparams.NChannels+bandMin[p*pieceLength+q]],sizeof(float)*(bandWidth[p]));
			#else
			memcpy(newWArrayPointer,&w[startSlice+i][p*pieceLength*sinoparams.NChannels+q*sinoparams.NChannels+bandMin[p*pieceLength+q]],sizeof(float)*(bandWidth[p]));
			memcpy(newEArrayPointer,&e[startSlice+i][p*pieceLength*sinoparams.NChannels+q*sinoparams.NChannels+bandMin[p*pieceLength+q]],sizeof(float)*(bandWidth[p]));
			#endif
			newWArrayPointer+=bandWidth[p];
			newEArrayPointer+=bandWidth[p];
		}
	}

	/* Initialize buffer for updated e (residual sinogram) */
	for (p = 0; p < NViewsdivided; p++)
	{
		#ifdef USE_INTEL_MEMCPY
		_intel_fast_memcpy(&CopyNewEArray[p][0],&newEArray[p][0],sizeof(float)*bandWidth[p]*pieceLength*SV_depth_modified);
		#else
		memcpy(&CopyNewEArray[p][0],&newEArray[p][0],sizeof(float)*bandWidth[p]*pieceLength*SV_depth_modified);
		#endif
	}

	/* Transpose the sinogram buffer so data relevant to a particular voxel update is fully contguous */
	float ** newWArrayTransposed = (float **)malloc(sizeof(float *) * NViewsdivided);
	float ** newEArrayTransposed = (float **)malloc(sizeof(float *) * NViewsdivided);

	for (p = 0; p < NViewsdivided; p++)
	{
		newWArrayTransposed[p] = (float *)malloc(sizeof(float)*bandWidth[p]*pieceLength*SV_depth_modified);
		newEArrayTransposed[p] = (float *)malloc(sizeof(float)*bandWidth[p]*pieceLength*SV_depth_modified);
	}

	float *WTransposeArrayPointer=&newWArrayTransposed[0][0];
	float *ETransposeArrayPointer=&newEArrayTransposed[0][0];
	
	/* Form the tranposed buffer, where data in each block is now ordered by (slice_index, channel_index, view_index) */
	for (p = 0; p < NViewsdivided; p++)
	for(currentSlice=0;currentSlice<(SV_depth_modified);currentSlice++) 
	{
		WTransposeArrayPointer=&newWArrayTransposed[p][currentSlice*bandWidth[p]*pieceLength];
		ETransposeArrayPointer=&newEArrayTransposed[p][currentSlice*bandWidth[p]*pieceLength];
		newEArrayPointer=&newEArray[p][currentSlice*bandWidth[p]*pieceLength];
		newWArrayPointer=&newWArray[p][currentSlice*bandWidth[p]*pieceLength];
		for(q=0;q<bandWidth[p];q++)
		{
			#pragma vector aligned 
			for(t=0;t<pieceLength;t++)
			{
				ETransposeArrayPointer[q*pieceLength+t]=newEArrayPointer[bandWidth[p]*t+q];
				WTransposeArrayPointer[q*pieceLength+t]=newWArrayPointer[bandWidth[p]*t+q];
			}
		}
	}

	WTransposeArrayPointer=&newWArrayTransposed[0][0];
	ETransposeArrayPointer=&newEArrayTransposed[0][0];
	newEArrayPointer=&newEArray[0][0];
	float inverseNumber=1.0/255;

	for (p = 0; p < NViewsdivided; p++)
		free((void *)newWArray[p]);

	free((void **)newWArray);

	/* BEGIN: Loop over each voxel in SV and compute voxel-wise ICD updates */
	/* Each voxel-wise ICD update is based on minimizing a local quadratic surrogate function whose coefficients are denoted by (theta_2, theta_1) */
	for(i=0;i<countNumber;i++)
	{
		const short j_new=j_newCoordinate[i];      /*XW: get the voxel's x,y location*/
		const short k_new=k_newCoordinate[i];
		float tempV[SV_depth_modified];			  
		float tempProxMap[SV_depth_modified];
		float neighbors[SV_depth_modified][10];
		char zero_skip_FLAG[SV_depth_modified];
		float max=max_num_pointer[j_new*Nx+k_new]; /* the voxel-wise maximum value of A-matrix column */
		float THETA1[SV_depth_modified];
		float THETA2[SV_depth_modified];
		memset(&THETA1[0],0.0, sizeof(THETA1));
		memset(&THETA2[0],0.0, sizeof(THETA2));	
		float diff[SV_depth_modified];

		int theVoxelPosition=(j_new-jy)*(2*SVLength+1)+(k_new-jx); 
		unsigned char * A_padd_Tranpose_pointer = &A_Padded_Map[theSVPosition][theVoxelPosition].val[0];

		/* for a given (jy,jx) coordinate, consecutively update all voxels along the slice-direction (jz) since we can re-use same A-matrix entries */
		for(currentSlice=0;currentSlice<SV_depth_modified;currentSlice++)
		{
			tempV[currentSlice] = (float)(image[startSlice+currentSlice][j_new*Nx+k_new]); /*XW: current voxel's value*/

			zero_skip_FLAG[currentSlice] = 0;

			/*if(reconparams.ReconType == MBIR_MODULAR_RECONTYPE_QGGMRF_3D)*/
			if(MBIRMode == MBIR_MAP_ESTIMATION)
			{
				ExtractNeighbors_WithinSlice(&neighbors[currentSlice][0],k_new,j_new,&image[startSlice+currentSlice][0],imgparams);

				if((startSlice+currentSlice)==0)
					neighbors[currentSlice][8]=voxelsBuffer1[j_new*Nx+k_new];
				else
					neighbors[currentSlice][8]=image[startSlice+currentSlice-1][j_new*Nx+k_new];

				if((startSlice+currentSlice)<(Nz-1))
					neighbors[currentSlice][9]=image[startSlice+currentSlice+1][j_new*Nx+k_new];
				else
					neighbors[currentSlice][9]=voxelsBuffer2[j_new*Nx+k_new];

				if (tempV[currentSlice] == 0.0)
				{
					zero_skip_FLAG[currentSlice] = 1;
					for (j = 0; j < 10; j++)
					{
						if (neighbors[currentSlice][j] != 0.0)
						{
							zero_skip_FLAG[currentSlice] = 0;
							break; 
						}
					}
				}
			}
			//if(reconparams.ReconType == MBIR_MODULAR_RECONTYPE_PandP)
			if(MBIRMode == MBIR_PnP_PRIORS)
				tempProxMap[currentSlice] = proximalmap[startSlice+currentSlice][j_new*Nx+k_new];
		}

		A_padd_Tranpose_pointer = &A_Padded_Map[theSVPosition][theVoxelPosition].val[0];
		/* loop over each block in the sinogram-buffer */
		for(p=0;p<NViewsdivided;p++)
		{
			/* For voxel-update get starting channel-index in the buffer and no. of channels */
			const int myCount=A_Padded_Map[theSVPosition][theVoxelPosition].pieceWiseWidth[p];
			const int pieceMin=A_Padded_Map[theSVPosition][theVoxelPosition].pieceWiseMin[p];
			/* update voxels along slice-direction */
			#pragma vector aligned
			for(currentSlice=0;currentSlice<SV_depth_modified;currentSlice++)
			if(zero_skip_FLAG[currentSlice] == 0 )
			{
				/* Point to sinogram-buffer at block-index p and slice-index  "currentSlice" */
				WTransposeArrayPointer=&newWArrayTransposed[p][currentSlice*bandWidth[p]*pieceLength];
				ETransposeArrayPointer=&newEArrayTransposed[p][currentSlice*bandWidth[p]*pieceLength];
				/* Move pointer to starting (minimum) channel for this voxel update */
				WTransposeArrayPointer+=pieceMin*pieceLength;
				ETransposeArrayPointer+=pieceMin*pieceLength;
				float tempTHETA1=0.0;
				float tempTHETA2=0.0;
				/* Accumulate and sum operation to compute theta_1 and theta_2 based on likelihood model for voxel update */     
				#pragma vector aligned
				#pragma simd reduction(+:tempTHETA2,tempTHETA1)
				for(t=0;t<myCount*pieceLength;t++)
				{	
					tempTHETA1 += A_padd_Tranpose_pointer[t]*WTransposeArrayPointer[t]*ETransposeArrayPointer[t];
					tempTHETA2 += A_padd_Tranpose_pointer[t]*WTransposeArrayPointer[t]*A_padd_Tranpose_pointer[t];
				}
				THETA1[currentSlice]+=tempTHETA1;
				THETA2[currentSlice]+=tempTHETA2;
			}
			A_padd_Tranpose_pointer+=myCount*pieceLength;
		}
		/* Scaling to account for 8-bit quantization of A-matrix */
		for(currentSlice=0;currentSlice<SV_depth_modified;currentSlice++)
		{
			THETA1[currentSlice]=-THETA1[currentSlice]*max*inverseNumber;
			THETA2[currentSlice]=THETA2[currentSlice]*max*inverseNumber*max*inverseNumber;
		}

		ETransposeArrayPointer=&newEArrayTransposed[0][0];

		A_padd_Tranpose_pointer = &A_Padded_Map[theSVPosition][theVoxelPosition].val[0];
	
		/* Increment theta_1 and theta_2 for voxel update based on prior model */
		for(currentSlice=0;currentSlice<SV_depth_modified;currentSlice++)
		if(zero_skip_FLAG[currentSlice] == 0)
		{
			float pixel,step;
			//if(reconparams.ReconType == MBIR_MODULAR_RECONTYPE_QGGMRF_3D)
			if(MBIRMode == MBIR_MAP_ESTIMATION)
			{
				QGGMRF3D_Update(reconparams.priorparams,tempV[currentSlice],&neighbors[currentSlice][0],&THETA1[currentSlice],&THETA2[currentSlice]);
			}
			else if(MBIRMode == MBIR_PnP_PRIORS)
			{
				PandP_Update(reconparams.SigmaPnPsq,tempV[currentSlice],tempProxMap[currentSlice],&THETA1[currentSlice],&THETA2[currentSlice]);
			}
			else
			{
				fprintf(stderr,"Error** Unrecognized MBIRMode in ICD update\n");
				exit(-1);
			}

			step  = -THETA1[currentSlice]/THETA2[currentSlice];
			pixel = tempV[currentSlice] + step;  /* can apply over-relaxation to the step size here */

			if(PositivityFlag)
				image[startSlice+currentSlice][j_new*Nx+k_new] = ((pixel < 0.0) ? 0.0 : pixel);
			else
				image[startSlice+currentSlice][j_new*Nx+k_new] = pixel;

			diff[currentSlice] = image[startSlice+currentSlice][j_new*Nx+k_new] - tempV[currentSlice];

			totalChange_loc += fabs(diff[currentSlice]);
			totalValue_loc += tempV[currentSlice];
			NumUpdates_loc++;

			diff[currentSlice]=diff[currentSlice]*max*inverseNumber;
		}

		/* Update local copy of residual-sinogram (e) buffer on each core */
		for(p=0;p<NViewsdivided;p++)
		{
			const int myCount=A_Padded_Map[theSVPosition][theVoxelPosition].pieceWiseWidth[p];
			const int pieceMin=A_Padded_Map[theSVPosition][theVoxelPosition].pieceWiseMin[p]; 
			#pragma vector aligned
			for(currentSlice=0;currentSlice<SV_depth_modified;currentSlice++)
			if(diff[currentSlice]!=0 && zero_skip_FLAG[currentSlice] == 0)
			{
				ETransposeArrayPointer=&newEArrayTransposed[p][currentSlice*bandWidth[p]*pieceLength];
				ETransposeArrayPointer+=pieceMin*pieceLength;

				#pragma vector aligned
				for(t=0;t<(myCount*pieceLength);t++)
					ETransposeArrayPointer[t]= ETransposeArrayPointer[t]-A_padd_Tranpose_pointer[t]*diff[currentSlice];
			}
			A_padd_Tranpose_pointer+=myCount*pieceLength;
		}
	}/* END: loop over each voxel in SV*/

	for (p = 0; p < NViewsdivided; p++)
		free((void *)newWArrayTransposed[p]);

	free((void **)newWArrayTransposed);

	/* Insert update-value for SV into heap */
	if(!NH_flag)
		headNodeArray[jj].x=totalChange_loc;
	else
		headNodeArray[indexList[jj]].x=totalChange_loc;

	/* Revert the transpose operation on the local e (updated residual-sinogram) buffer, so that data order is (slice_index, view_index, channel_index). */
	for (p = 0; p < NViewsdivided; p++)
	for(currentSlice=0;currentSlice<SV_depth_modified;currentSlice++)
	{
		ETransposeArrayPointer=&newEArrayTransposed[p][currentSlice*bandWidth[p]*pieceLength];
		newEArrayPointer=&newEArray[p][currentSlice*bandWidth[p]*pieceLength]; 
		for(q=0;q<bandWidth[p];q++)
		{
			#pragma vector aligned
			for(t=0;t<pieceLength;t++)
				newEArrayPointer[bandWidth[p]*t+q]=ETransposeArrayPointer[q*pieceLength+t];
		}
	}

	for (p = 0; p < NViewsdivided; p++)
		free((void *)newEArrayTransposed[p]);

	free((void **)newEArrayTransposed);

	newEArrayPointer=&newEArray[0][0];
	float* CopyNewEArrayPointer=&CopyNewEArray[0][0];
	float* eArrayPointer=&e[0][0];

	/* Update global e data based on local e (residual-sinogram) buffer */
	for (p = 0; p < NViewsdivided; p++)     
	{
		newEArrayPointer=&newEArray[p][0];
		CopyNewEArrayPointer=&CopyNewEArray[p][0];
		for (currentSlice=0; currentSlice< SV_depth_modified;currentSlice++)
		{
			#pragma vector aligned
			for(q=0;q<pieceLength;q++)
			{
				eArrayPointer=&e[startSlice+currentSlice][p*pieceLength*sinoparams.NChannels+q*sinoparams.NChannels+bandMin[p*pieceLength+q]];
				for(t=0;t<bandWidth[p];t++)
				{
					#pragma omp atomic
					*eArrayPointer += (*newEArrayPointer)-(*CopyNewEArrayPointer); 
					newEArrayPointer++;
					CopyNewEArrayPointer++;
					eArrayPointer++;
				}
			}
		}
	}

	for (p = 0; p < NViewsdivided; p++)
	{
		free((void *)newEArray[p]);
		free((void *)CopyNewEArray[p]);
	}
	free((void **)newEArray);
	free((void **)CopyNewEArray);

	*NumUpdates += NumUpdates_loc;
	*totalValue += totalValue_loc;
	*totalChange += totalChange_loc;

}   /* END super_voxel_recon() */




void coordinateShuffle(int *order1, int *order2,int len)
{
	int i, j, tmp1,tmp2;

	for (i = 0; i < len-1; i++)
	{
		j = i + (rand() % (len-i));
		tmp1 = order1[j];
		tmp2 = order2[j];
		order1[j] = order1[i];
		order2[j] = order2[i];
		order1[i] = tmp1;
		order2[i] = tmp2;
	}
}

void three_way_shuffle(int *order1, int *order2,struct heap_node *headNodeArray,int len)
{
	int i, j, tmp1,tmp2;

	float temp_x;

	for (i = 0; i < len-1; i++)
	{
		j = i + (rand() % (len-i));
		tmp1 = order1[j];
		tmp2 = order2[j];
		temp_x=headNodeArray[j].x;
		order1[j] = order1[i];
		order2[j] = order2[i];
		headNodeArray[j].x=headNodeArray[i].x;
		order1[i] = tmp1;
		order2[i] = tmp2;
		headNodeArray[i].x=temp_x;
	}
}



float MAPCostFunction3D(float **e,struct Image3D *Image,struct Sino3DParallel *sinogram,struct ReconParams *reconparams)
{
	int i, M, j, Nx, Ny, Nz;
    float **x ;
    float **w ;
    float nloglike;

    x = Image->image;
    w = sinogram->weight;
    
	M = sinogram->sinoparams.NViews * sinogram->sinoparams.NChannels ;
	Nx = Image->imgparams.Nx;
	Ny = Image->imgparams.Ny;
    Nz = Image->imgparams.Nz;
    
	nloglike = 0.0;
	for (i = 0; i <sinogram->sinoparams.NSlices; i++){
	for (j = 0; j < M; j++)
    	nloglike += e[i][j]*w[i][j]*e[i][j];
    }

	nloglike /= 2.0;
	
	return (nloglike + QGGMRFCostFunction3D(x, reconparams->priorparams, Image->imgparams)) ;
}

/* Separated this out since QGGMRF can be used in plug-n-play mode */
float QGGMRFCostFunction3D(float **x, struct PriorParams priorparams, struct ImageParams3D imgparams)
{
	int j,Nx,Ny,Nz,jx,jy,jz,plusx,minusx,plusy,plusz;
	float nlogprior_nearest = 0.0, nlogprior_diag = 0.0, nlogprior_interslice = 0.0;

	Nx=imgparams.Nx;
	Ny=imgparams.Ny;
	Nz=imgparams.Nz;

	for (jz = 0; jz < Nz; jz++)
	for (jy = 0; jy < Ny; jy++)
	for (jx = 0; jx < Nx; jx++)
	{
		plusx = jx + 1;
		plusx = ((plusx < Nx) ? plusx : 0);
		minusx = jx - 1;
		minusx = ((minusx < 0) ? Nx-1 : minusx);
		plusy = jy + 1;
		plusy = ((plusy < Ny) ? plusy : 0);
		plusz = jz + 1;
		plusz = ((plusz < Nz) ? plusz : 0);

		j = jy*Nx + jx; 

		nlogprior_nearest += QGGMRF_Potential((x[jz][j] - x[jz][jy*Nx+plusx]), priorparams);
		nlogprior_nearest += QGGMRF_Potential((x[jz][j] - x[jz][plusy*Nx+jx]), priorparams);

		nlogprior_diag += QGGMRF_Potential((x[jz][j] - x[jz][plusy*Nx+minusx]),priorparams);
		nlogprior_diag += QGGMRF_Potential((x[jz][j] - x[jz][plusy*Nx+plusx]), priorparams);

		nlogprior_interslice += QGGMRF_Potential((x[jz][j] - x[plusz][jy*Nx+jx]),priorparams);
	}

	return(priorparams.b_nearest * nlogprior_nearest + priorparams.b_diag * nlogprior_diag + priorparams.b_interslice * nlogprior_interslice);
}


void MannUpdate( float **img_current, float **img_previous, float Rho, struct ImageParams3D *imgparams)
{
	int jz, jyx, Nz, Nyx;
	Nz  = imgparams->Nz;
	Nyx = imgparams->Nx * imgparams->Ny;

	for(jz=0;jz<Nz;jz++)
	for(jyx=0;jyx<Nyx;jyx++)
		img_current[jz][jyx] = Rho*img_current[jz][jyx] + (1-Rho)*img_previous[jz][jyx];

}


void Reflect(float **img_reflected, float **img_in, float **img_ref, struct ImageParams3D *imgparams)
{
	int jz, jyx, Nz, Nyx;
	Nz  = imgparams->Nz;
	Nyx = imgparams->Nx * imgparams->Ny;

	for(jz=0;jz<Nz;jz++)
	for(jyx=0;jyx<Nyx;jyx++)
		img_reflected[jz][jyx] = 2.0*img_in[jz][jyx] - img_ref[jz][jyx] ;
}


#if 0
// NOTE this needs to be updated
void read_golden(char *fname,float **golden,int Nz,int N, struct Image3D *Image)
{
	FILE *fp;
	int i;
        char slicefname[200];
        char *sliceindex;
		sliceindex= (char *)malloc(MBIR_MODULAR_MAX_NUMBER_OF_SLICE_DIGITS);
		for(i=0;i<Nz;i++){
			sprintf(sliceindex,"%.*d",MBIR_MODULAR_MAX_NUMBER_OF_SLICE_DIGITS,Image->imgparams.FirstSliceNumber+i);

		/* Obtain file name for the given slice */
		strcpy(slicefname,fname);
		strcat(slicefname,"_slice"); 
		strcat(slicefname,sliceindex); /* append slice index */
		strcat(slicefname,".2Dimgdata");
		if ((fp = fopen(slicefname, "r")) == NULL)
		{
			fprintf(stderr, "ERROR in read golden: can't open file %s.\n", slicefname);
			exit(-1);
		}

		fread(&golden[i][0],sizeof(float),N,fp);
		fclose(fp);
	}

}
#endif

#if 0
static __inline__ unsigned long long rdtsc()
{
   unsigned hi,lo;
   __asm __volatile__ ("rdtsc" : "=a"(lo),"=d"(hi));
   return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
#endif




