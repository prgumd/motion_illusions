%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% expspir.m
%
% Based on expir.c which was used for the paper:
% Families of stationary patterns producing illusory movement: insights into the visual system
% Authors: Fermuller, Pless, and Aloimonos
%
% A major difference is solid bands are drawn using two tangential arcs instead
% of a fixed width line
%
% Levi Burner
%
% History:
% 01-23-20 - Created File
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Good defaults are
% LINES = 120
% RAY_WIDTH = 0.4
% BANDS = 7
% BAND_WIDTH = 0.1
% K = -1.2 (tightness of spiral)
function [wedges, bands] = expspir(LINES, RAY_WIDTH, BANDS, BAND_WIDTH, K)
    LINES = floor(LINES);
    BANDS = floor(BANDS);

    wedges = polyshape.empty(0, 0);
    bands = polyshape.empty(0, 0);

    for i = 0:LINES-1
        th = i * 2 * pi / LINES;

        wedge_points = [exp_spiral_path(th, K);
                        flip(exp_spiral_path(th + RAY_WIDTH * 2 * pi / LINES, K))];

        wedges(end+1) = polyshape(wedge_points);
    end

    for i = 0:BANDS-1
        th = i * 2 * pi / BANDS;

        band_points = [tangent_exp_spiral_path(th, K);
                       flip(tangent_exp_spiral_path(th + BAND_WIDTH * 2 * pi / BANDS, K))];

        bands(end+1) = polyshape(band_points);
    end
end

function path = exp_spiral_path(th, K)
    x = 0;
    y = 0;
    path = double.empty(0, 2);
    path = add_point(x, y, path);

    for j = 0:0.03:6-0.03
        r = j;
        x = exp(r) * cos(K * r + th);
        y = exp(r) * sin(K * r + th);
        path = add_point(x, y, path);
    end
end

function path = tangent_exp_spiral_path(th, K)
    x = 5 * cos(th);
    y = 5 * sin(th);
    path = double.empty(0, 2);
    path = add_point(x, y, path);

    while sqrt(x * x + y * y) < 200
        expr = sqrt(x * x + y * y);

        r = log(expr);

        if x > 0
            if y > 0 th = asin( y / expr ) - K * r; end
            if y == 0 th = 0; end
            if y < 0 th = (2 * pi - asin( -y / expr)) - K * r; end
        else
            if (y > 0) th = (pi / 2 + asin(-x / expr)) - K * r; end
            if (y == 0) th = -pi; end
            if (y < 0) th = (pi + asin(-y/expr)) - K * r; end
        end

        dx = K * expr * cos(K * r + th) + expr *sin(K * r + th);
        dy = - ( expr * cos(K * r + th) - expr * K *  sin(K * r + th));

        STEPSIZE = 0.01;
        if (x * dx + y * dy > 0)
          x = x + STEPSIZE * dx;
          y = y + STEPSIZE * dy;
        else
          x = x - STEPSIZE * dx;
          y = y - STEPSIZE * dy;
        end

        path = add_point(x, y, path);
    end
end

function poly_points = add_point(x, y, poly_points)
    % SCALE is copied from the original expspir.c implementation
    % scale is the equivalent scale for the matlab window coordinates
    SCALE = 2.0;
    scale = 2*(SCALE/(508-68));
    poly_points(end+1, :) = [x, y] * scale;
end
