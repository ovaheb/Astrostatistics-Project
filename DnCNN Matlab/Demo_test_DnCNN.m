
%%% This is the testing demo for gray image (Gaussian) denoising.
%%% Training data: 400 images of size 180X180

clear;
clc;
addpath('utilities');
%folderTest  = fullfile('testsets','Set12'); %%% test dataset
folderTest  = fullfile('testsets','SET12_Dataset_Noisy/gt'); %%% test dataset
folderModel = 'model';
noiseSigma  = 1;  %%% image noise level
showResult  = 1;
useGPU      = 0;
pauseTime   = 0;
folderNoise = 'NoiseLevel_' + string(noiseSigma);

%%% load [specific] Gaussian denoising model

%modelSigma  = min(75,max(10,round(noiseSigma/5)*5)); %%% model noise level
%load(fullfile(folderModel,'specifics',['sigma=',num2str(modelSigma,'%02d'),'.mat']));

%%% load [blind] Gaussian denoising model %%% for sigma in [0,55]

load(fullfile(folderModel,'GD_Gray_Blind.mat'));


%%%
%net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

%%% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));
tic
for i = 1:length(filePaths)
    
    %%% read images
    label = imread(fullfile(folderTest,filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    label = im2double(label);
    
    %randn('seed',0);

    newStr = replace(fullfile(folderTest,filePaths(i).name),'gt',folderNoise);
    input = imread(newStr);
    input = im2double(input);
    input = single(input);

    %res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
    output = input - res(end).x;
    
    %%% calculate PSNR and SSIM
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showResult
        imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        drawnow;
        pause(pauseTime)
    end
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
end
toc

disp([mean(PSNRs),mean(SSIMs)]);
disp(tic, toc)