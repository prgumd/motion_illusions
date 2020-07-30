clear all;
clc;
close all;

% Simulator Parameters
t = 0.001;
delta_t = 0.001;
tol = 0.001;
C = 0.15;
event_array = [];
eps = 0.01;


fname1 = sprintf('sample_images/%04d.png', t*1000);
I1 = double(rgb2gray(imread(fname1)));
reference_values = log(eps+I1);

for i = 25 :1:100
    i
    event_array = [];
    fname1 = sprintf('sample_images/%04d.png', i);
    fname2 = sprintf('sample_images/%04d.png', i+1);
    I1 = double(rgb2gray(imread(fname1)));
    I2 = double(rgb2gray(imread(fname2)));
    I1log = log(eps+I1);
    I2log = log(eps+I2);

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
        axis([0 640 0 300 -inf inf])
    %     axis equal tight
        xlabel('X')
        ylabel('Y')
        zlabel('Time (s)')
        hold on;
    

    dlmwrite('events.txt' ,event_array, '-append');

    plot3 (event_array(:,1), event_array(:,2), event_array(:,3),'.') 
    hold on;
    view(0,90)
    pause;
    clf('reset')
end
