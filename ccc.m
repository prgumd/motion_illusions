%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ccc.m
%
% Based on ccc_update.c which was used for the paper:
% Families of stationary patterns producing illusory movement: insights into the visual system
% Authors: Fermuller, Pless, and Aloimonos
%
% The port is not complete.
%
% Levi Burner
%
% History:
% 01-21-20 - Created File
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [wedges, bands] = ccc(LINES)
    LINES = floor(LINES);
    SCALE = 120;
    CENTER = 288;
    RAY_WIDTH = 0.01;
    BAND_WIDTH = 20.0;
    BANDS = 8;
    STEPFACTOR = 0.003;
    STEPS = 4000;
    BOFFSET = 0.3;

    wedges = polyshape.empty(0, 0);
    bands = polyshape.empty(0, 0);

    k = 0;
    a = 0;
    b = 0;
    f = 0;
    m = 0;
    BB = 0;

    dx = 0;
    dy = 0;
    C = 0;
    x = 0;
    y = 0;

    stepsize = 0.005;
    dist = 0;
    temp = 0;

    a = 0.2;
    b = 0.3;
    f = 1;
    m = sqrt(a^2 + b^2);

    ADD_POINT = @(x, y, poly_x, poly_y) (add_point(x, y, SCALE, CENTER, poly_x, poly_y));

    for i = (-CENTER/(115*m)) : ((1 + CENTER/((115) * m)) /  LINES) : 1
        poly_x = [];
        poly_y = [];

        x = i * a;
        y = i * b;

        [poly_x, poly_y] = ADD_POINT(x, y, poly_x, poly_y);

        for j = 0:STEPS-1
            dx = -( -a * (y*y + f*f) + b * x * y + 1 * x * f);
            dy = -(a * x * y - b * (x*x + f*f) + 1 * y * f);

            % move along the tangent lines
            dist = sqrt( (x - a) * (x-a) + (y - b) * (y-b));

            stepsize = STEPFACTOR * dist / sqrt(dx*dx + dy*dy);

            x = x - dy * stepsize;
            y = y + dx * stepsize;

            if mod(j, 20) == 0
                [poly_x, poly_y] = ADD_POINT(x, y, poly_x, poly_y);
            end

            if (abs(x) > CENTER / 115) || (abs(y) > CENTER / 115)
                break;
            end
            
            if -b * x + a * y < -0.001
                break;
            end
        end

        [poly_x, poly_y] = ADD_POINT(x, y, poly_x, poly_y);

        i = i + (0.60 * ((1 + CENTER/((115) * m)) /  LINES));
        x = i * a;
        y = i * b;

        for j = 0:STEPS-1
            dx = -( -a * (y*y + f*f) + b * x * y + 1 * x * f);
            dy = -(a * x * y - b * (x*x + f*f) + 1 * y * f);
            % move along the tangent lines
            dist = sqrt( (x - a) * (x-a) + (y - b) * (y-b));
            stepsize = STEPFACTOR * dist / sqrt(dx*dx + dy*dy);
            x = x - dy * stepsize;
            y = y + dx * stepsize;
            if ( (abs(x) > CENTER / (115)) || (abs(y) > CENTER / (115)))
                break;
            end
            if ( -b * x + a * y < -0.001 ) 
                break;
            end
        end

        [poly_x, poly_y] = ADD_POINT(x, y, poly_x, poly_y);

        for j = STEPS-1:-1:1
            dx = ( -a * (y*y + f*f) + b * x * y + 1 * x * f);
            dy = (a * x * y - b * (x*x + f*f) + 1 * y * f);

            % move along the tangent lines
            dist = sqrt( (x - a) * (x-a) + (y - b) * (y-b));

            stepsize = STEPFACTOR * dist / sqrt(dx*dx + dy*dy);
            x = x - dy * stepsize;
            y = y + dx * stepsize;

            if ( (abs(x) > CENTER / (115)) || (abs(y) > CENTER / (115)))
                break;
            end

            if mod(j, 20) == 0
                [poly_x, poly_y] = ADD_POINT(x, y, poly_x, poly_y);
            end

            if ((-b *x + a * y < -0.001) && ((x * i * a + y * i * b) > 0))
                break;
            end
        end

        [poly_x, poly_y] = ADD_POINT(x, y, poly_x, poly_y);

        wedges(end+1) = polyshape(poly_x, poly_y);
    end
end

function [poly_x, poly_y] = add_point(x, y, SCALE, CENTER, poly_x, poly_y)
    poly_x(end+1) = x * 2*(SCALE/(508-68));
    poly_y(end+1) = y * 2*(SCALE/(508-68));
end
