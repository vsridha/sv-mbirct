/* ============================================================================== 
 * Copyright (c) 2016 Xiao Wang (Purdue University)
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

#ifndef _PNPDENOISER_H_
#define _PNPDENOISER_H_

void Proximal_QGGMRF_Denoising(struct Image3D *Image, struct PriorParams priorparams, char *ImageReconMask, struct Image3D *ProximalMapInput, \
							   float  SigmaPnP, int NumICDPasses, char   positivity_flag);


float ProximalMapCostFunction3D_QGGMRF(struct Image3D *Image, struct PriorParams priorparams, struct Image3D *ProximalMapInput, float  SigmaPnP);

void RandomizedICD_QGGMRF(struct Image3D *Image, struct Image3D *ProximalMapInput, char *ImageReconMask, struct PriorParams priorparams, \
						  float SigmaPnP, int *order, int Nyx, int Nz, char positivity_flag, float *avg_change) ;


void BM3DDenoise(struct Image3D *CleanImage, struct Image3D *NoisyImage, struct PriorParams priorparams); /* Input - noisy image. Output - clean (denoised) image */

void CNNDenoise(struct Image3D *CleanImage, struct Image3D *NoisyImage, struct PriorParams priorparams);  /* Input - noisy image. Output - clean (denoised) image */

#endif
