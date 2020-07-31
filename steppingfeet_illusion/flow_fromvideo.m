clear all;
clc;
close all;

% Video reader and optical flow initializer
vidReader = VideoReader('bar_gradual.avi');
opticFlow = opticalFlowLK('NoiseThreshold',0.009);

% Output visualization parameters
flowvector_vis = 1; % Flow Vector Visualization: 1 = True, 0 = False
background = 1; % Output background options: 1 = Input frame, 2 = Flow color wheel, 0 = Blank image

h = figure;
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);
blankimage = ones(vidReader.height, vidReader.width, 3);


while hasFrame(vidReader)
    frameRGB = readFrame(vidReader);
    frameGray = rgb2gray(frameRGB);
    flow = estimateFlow(opticFlow,frameGray);
   
    if background == 1
        imshow(frameRGB)
    elseif background == 2
         flowimg = flowToColor(cat(3,flow.Vx, flow.Vy));
         imshow(flowimg)
    else
         imshow(blankimage)
    end
    if flowvector_vis == 1
        hold on
        plot(flow,'DecimationFactor',[5 5],'ScaleFactor',10,'Parent',hPlot);
        hold off
    end
    pause(10^-3)
end