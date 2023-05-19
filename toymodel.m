%create toy supercell pattern
clear
close all
motif_type='tri';%toy motif provided: 'tri', 'diamond' or 'sandglass'
supercell=round(double(imread([motif_type,'.tiff']))/256);
x1=0.5*linspace(-pi,pi,60);
y1=0.5*linspace(-pi,pi,51);
[X1,Y1]=meshgrid(x1,y1);
supercell(:,:,1)=[];
supercell=exp(1j*Y1.*supercell);
sppattern=angle(exp(-1j*angle(supercell)).*exp(1j*circshift(angle(supercell),1,1)));%normal strain (NS)
sppattern=sppattern(2:end,:);
imagesc(sppattern)
caxis([-0.05,0])
title('Supercell Pattern')
clear x1 X1 y1 Y1 sppattern
%%
%simulate diffraction pattern
diffpatternC=zeros(1500,2000);
h=waitbar(0,'Please Wait');
iter=1000;
for i=1:iter
    waitbar(i/iter,h)
    toyobject=supermix(1500,2000,25,12,14,150,supercell);
    diffpattern=abs(fftshift(ifft2(ifftshift(toyobject)))).^2;
    diffpatternC=diffpatternC+diffpattern;
end
delete(h)
figure
imagesc(log10(diffpatternC))
title('Simulated Diffraction Pattern')
clear iter toyobject diffpattern i h
%%
%initialization
data=diffpatternC;
data=data/max(data(:))*1e5;
mean1=mean(data,1);
[M1,I1]=max(mean1);
mean2=mean(data(:,I1-2:I1+2),2);
[M2,I2]=max(mean2);
QrangeHor=250;
QrangeVer=[220];
srhor=80;
srver=7;
usegpu=true;%if the reconstruction use gpu or not

figure
data=data(I2-QrangeVer:I2+QrangeVer-1,I1-QrangeHor:I1+QrangeHor-1);
imagesc(log10(data))
title('Input Diffraction Pattern')
clear mean1 M1 I1 mean2 M2 I2 diffpatternC
%%
% Define Support & Initialization
support=zeros(size(data));
[sx,sy]=meshgrid(-QrangeHor:QrangeHor-1,-QrangeVer:QrangeVer-1);
support((abs(sx)<srhor)&(abs(sy)<srver))=1;

G0=sqrt(data);
G0=ifftshift(G0);
support=ifftshift(support);
Simnum=200;%number of random starts/simulations;
Collector=zeros([size(data),Simnum]);

%change matrices to gpu
if usegpu
    Collector=gpuArray(Collector);
    G0=gpuArray(G0);
    support=gpuArray(support);
end
clear sx sy
%%
% CDI
for nn=1:Simnum
    nn
    rng shuffle
    gk=exp(1j*2*pi*rand(size(data)));%random start
    if usegpu
        gk=gpuArray(gk);
    end
    for n=1:1999 %number of iteration
        G0c=G0;
        
        if mod(n,100)<60
            gk=OOPCSYM(gk,G0c,support,0.6);
        elseif mod(n,100)<80
            gk=OOPC(gk,G0c,support,0.8);
        else
            gk=OOPC(gk,G0c,support,0.98);
        end
    end
    clear Gk Gkp
    if usegpu
        Collector(:,:,nn)=gather(gk).*exp(-1j*angle(gather(gk(1,1))));
    else
        Collector(:,:,nn)=gk.*exp(-1j*angle(gk(1,1)));
    end
end
support=fftshift(support);
G0=fftshift(G0);
if usegpu
    [Collector,G0,support]=gather(Collector,G0,support);
end
clear nn gk G0c n
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%clustering part%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Collectorshift=zeros(2*srver-1,2*srhor-1);
for i=1:size(Collector,3)
    a=fftshift(squeeze(Collector(:,:,i)));
    Collectorshift(:,:,i)=a(size(data,1)/2-srver+2:size(data,1)/2+srver,size(data,2)/2-srhor+2:size(data,2)/2+srhor);
end

C21=Collectorshift;
for i=1:size(C21,3)
    a=angle(C21(:,:,i));
    b=angle(C21(:,:,i));
    a=angle(exp(-1j*a).*exp(1j*circshift(a,1,1)));%normal strain (NS)
    C22(:,:,i)=a(2:end,:);
    
    b=angle(exp(1j*b).*exp(-1j*circshift(b,1,2)));%crystal plane inclination (CI)
    C23(:,:,i)=b(:,2:end);
end

clear a b C21 Collectorshift i
%%
%autocorrelation of recs
figure
autocorr=zeros(size(C22,1)*2-1,size(C22,2)*2-1);
for i=1:size(Collector,3)
    autocorr=autocorr+xcorr2(C22(:,:,i),C22(:,:,i));
end
plot(sum(autocorr))
title('Reconstructions AutoCorrelation')


p=17;%period length gained from the autocorrelation peak distance
sp=-sin(2*pi/p*([0:p-1])-pi/2);%sinusoidal test function
clear autocorr i
%%
%cross correlate the peak with recs
uccollector=zeros(size(C22,1),p);
uccollector_CI=zeros(size(C23,1),p-1);
for i=1:size(Collector,3)
    i
    recprofile=sum(C22(:,:,i),1);
    crscor=xcorr(recprofile,sp);
    crscor=crscor(159:301);
    [pks,locs]=findpeaks(crscor/max(crscor),'MinPeakDistance',10,'MinPeakHeight',0.3);
    for j=1:size(locs,2)
        uccollector=cat(3,uccollector,C22(:,locs(j):locs(j)+p-1,i));%supercell NS collector
        uccollector_CI=cat(3,uccollector_CI,C23(:,locs(j):locs(j)+p-2,i));%supercell CI collector
    end
end
uccollector(:,:,1)=[];
uccollector_CI(:,:,1)=[];
clc
clear i crscor pks locs j recprofile
%%
%normalization for NS supercell
C6=reshape(uccollector,[size(uccollector,1)*size(uccollector,2),size(uccollector,3)]);
C62=C6-mean(C6,2)*ones(1,size(C6,2));
C63=ones(size(C62));
for i=1:size(C62,2)
    C63(:,i)=C62(:,i)/norm(C62(:,i));
end
clear C6 C62 i
%%
%kmeans clustering
idxc=cell(8,1);
for clusternum=2:8
    clusternum
    C=C63;
    struct_stats=statset('UseParallel',true,'MaxIter',10000);
    [idx,C,sumd,D]=kmeans(C',clusternum,'Replicates',1000,'Options',struct_stats);
    idxc{clusternum}=idx;
end
clear clusternum C C63 idx sumd D struct_stats
%%
close all

for clusternum=2:8
    for i=1:clusternum
        clusterindex=find(idxc{clusternum}==i);

        figure(1)
        subplot(ceil(clusternum/2),2,i)
        imagesc(squeeze(sum(uccollector_CI(:,:,clusterindex),3))/numel(clusterindex))
        title(['CI Cluster#',num2str(i)])
        axis tight
        axis equal
        
        figure(2)
        subplot(ceil(clusternum/2),2,i)
        imagesc(squeeze(sum(uccollector(:,:,clusterindex),3))/numel(clusterindex))
        title(['NS Cluster#',num2str(i)])
        axis tight
        axis equal
    end
    pause
end
clear clusternum i clusterindex
clc
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions


%Symmetrize the realspace pattern along the horizontal axis
function gkk=SYM(gkp)
gkk=abs(gkp).*exp(1j*(angle(gkp)+fliplr(angle(gkp)))/2);
end


% %Hibrid Output-Outpot algorithm
function gkk=OOPC(gk,G0,support,beta)
Gk=ifft2(gk);
Gkp=G0.*exp(1j*angle(Gk));%fourier space constrain
gkk=fft2(Gkp);
gkk=exp(1j*angle(gkk));
gkk=support.*gkk+(1-support).*(gkk-beta*gkk);
end

% %Hibrid Output-Outpot sym algorithm from python
function gkk=OOPCSYM(gk,G0,support,beta)
Gk=ifft2(gk);
Gkp=G0.*exp(1j*angle(Gk));%fourier space constrain
gkk=fft2(Gkp);
gkk=exp(1j*angle(gkk));
gkk=SYM(gkk);
gkk=support.*gkk+(1-support).*(gkk-beta*gkk);
end


function bg=supermix(Msize1,Msize2,Ssize1,pnummin,pnummax,Sgap,supercell)
%Msize1/2 size of entire matrix 1/2
%Ssize1 size of Support in direction 1
%pnum:number of periods in the support
%Sgap, the sum of random gaps in pixels
pnum=randi([pnummin,pnummax]);
Sgap=round(Sgap/pnummax*pnum);
bg=zeros(Msize1,Msize2);
id=floor(rand(pnum,1)*1);
igap=round(abs(randn(pnum,1))*2);
igap=floor(igap/sum(igap(:))*Sgap*pnum/(pnum+1));
igap(pnum+1,1)=Sgap-sum(igap(:));
c1=round(Msize1/2);
c2=round(Msize2/2);
Ssize2=pnum*size(supercell,2)+Sgap;

if mod(Ssize2,2)==0
    bg(c1-Ssize1:c1+Ssize1,c2-Ssize2/2:c2+Ssize2/2-1)=1;
else
    bg(c1-Ssize1:c1+Ssize1,c2-(Ssize2-1)/2:c2+(Ssize2-1)/2)=1;
end

for i=1:pnum
    pt=supercell;
    gapsum=sum(igap(1:i));
    bg(c1-Ssize1:c1+Ssize1,c2-floor(Ssize2/2)+gapsum+(i-1)*size(pt,2):c2-floor(Ssize2/2)+gapsum+(i)*size(pt,2)-1)=pt;
end
end
