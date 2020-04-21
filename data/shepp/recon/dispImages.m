% Display result of "shepp" demo

Nx=128;

fname='shepp_slice0001.2Dimgdata';

fp=fopen(fname,'r'); 
img=fread(fp,[Nx,Nx],'float32');
fclose(fp);

%figure(1); 
fig=figure('visible','off')
clf; 
colormap(gray(256))
imagesc(img',[0,0.04]); colorbar; 
title(fname,'interpreter','none')
axis('image')
print(fig,'shepp','-dpdf')


