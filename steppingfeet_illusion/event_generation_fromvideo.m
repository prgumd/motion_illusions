clear all;
clc;
close all;

% Simulator Parameters
t = 0.001;
delta_t = 0.001;
tol = 0.001;
C = 0.2; %0.15
event_array = [];
eps = 0.001;

% Video reader parameters
vidObj = VideoReader('bar_gradual.avi');
n = vidObj.NumberOfFrames;
frame_rate = vidObj.FrameRate;
duration = vidObj.Duration;
w = vidObj.Width;
h = vidObj.Height;

% Image to log Conversion 
frame1 = read(vidObj, 1);
vidObj.CurrentTime;
I1 = double(rgb2gray(frame1));
reference_values = log(eps+I1);


for i = 25 :1:n
    i
    event_array = [];
    I1 = rgb2gray(read(vidObj, i));
    I2 = rgb2gray(read(vidObj, i+1));
    
    I1log = log(eps+double(I1));
    I2log = log(eps+double(I2));

    [U V]= size(I1);
    event_array = [];
    for u = 1:U
        for v = 1:V

            It = I1log(u, v);
            It_dt = I2log(u, v);
            previous_crossing = reference_values(u,v);
            if(abs(It_dt - It)> tol)
                if (It_dt >= It); polarity = +1; else; polarity = -1; end

                list_crossings = [];
                all_crossings_found = 0;
                cur_crossing = previous_crossing;
                while (all_crossings_found ~= 1)
                    cur_crossing = cur_crossing + polarity * C;
                    
                    if polarity > 0
                        if (cur_crossing > It & cur_crossing <= It_dt)
                            list_crossings = [list_crossings;cur_crossing];
                        else
                            all_crossings_found = 1;
                        end
                    else
                        if (cur_crossing < It & cur_crossing >= It_dt)
                            list_crossings = [list_crossings;cur_crossing];
                        else
                            all_crossings_found = 1;
                        end
                    end

                    for i = 1: length(list_crossings)
                        te = t + (list_crossings(i)-It) * delta_t / (It_dt-It);
                        event = [v u te polarity]; 
                        event_array = [event_array; event];
                    end

                    if (length(list_crossings)>0)
                        reference_values(u,v) = list_crossings(1);
                    end
                end

            end
            
            
        end
    end
    
    t = t + delta_t;
    
    Idif = I2log - I1log;
        set(gcf,'numbertitle','off','name','Point2D')
        axis([0 w 0 h -inf inf])
    %     axis equal tight
        xlabel('X')
        ylabel('Y')
        zlabel('Time (s)')
        hold on;
    

    dlmwrite('events_exp.txt' ,event_array, '-append');

    plot3 (event_array(:,1), event_array(:,2), event_array(:,3),'.') 
    hold on;
    view(0,90)
    pause;
    clf('reset')
end
