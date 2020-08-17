%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% coaxis_spiral.m
%
% Create a spiral co-axis pattern
% Levi Burner
%
% History:
% 02-07-20 - Created File
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [coradial_shapes, coaxis_shapes] = coaxis_spiral(CORADIAL_SHAPES, CORADIAL_WIDTH, COAXIS_SHAPES, COAXIS_WIDTH, K, A, B, C)
    CORADIAL_SHAPES = floor(CORADIAL_SHAPES);
    COAXIS_SHAPES = floor(COAXIS_SHAPES);

    coradial_shapes = polyshape.empty(0, 0);
    coaxis_shapes = polyshape.empty(0, 0);

    % Normalize A, B, C to length 1
    absABC = norm([A, B, C]);
    A = A / absABC;
    B = B / absABC;
    C = C / absABC;
    f = 1;

    % Invert K so that coradial shapes spiral away from origin
    K = -K;

    for i = 0:CORADIAL_SHAPES-1
        th = i * 2 * pi / CORADIAL_SHAPES;
        %th_width = CORADIAL_WIDTH * (2 * pi / CORADIAL_SHAPES);

        coradial_shapes(end+1) = spiral_shape(th, CORADIAL_WIDTH, K, A, B, C, f, @perp_to_coaxis_vector);
    end

    for i = 0:COAXIS_SHAPES-1
        th = i * 2 * pi / COAXIS_SHAPES;
        %th_width = COAXIS_WIDTH * (2 * pi / COAXIS_SHAPES);

        coaxis_shapes(end+1) = spiral_shape(th, COAXIS_WIDTH, K, A, B, C, f, @coaxis_vector);
    end
end

function shape = spiral_shape(th, width, K, A, B, C, f, tangent_source)
    STOP_RADIUS = 2.0;
    MAX_POINTS = 10000;

    % Calculating the points spiraling away from the the coaxis vectors intersection with the image plane
    outgoing_coradial_points = spiral_path(th, K, A, B, C, f, tangent_source,...
                                           STOP_RADIUS,... % Radius to stop at between min and max points
                                           0, MAX_POINTS,... % min and max points
                                           0); % width

    % Limit the incoming number of points in hopes that the two paths end up finishing close to each other
    incoming_max_points = min(MAX_POINTS, length(outgoing_coradial_points));

    % Calculate another spiral but shift the start point by a given offset to create "width"
    % Flip the result so that it is spiraling towards the coaxis vectors intersection with the image plane
    % Make sure there are at least as many points as outgoing and at most as many points as outgoing
    incoming_coradial_points = flip(spiral_path(th, K, A, B, C, f, tangent_source,...
                                                STOP_RADIUS,... % Radius to stop at between min and max points
                                                length(outgoing_coradial_points), incoming_max_points,... % min and max points
                                                width)); % width

    coradial_points = [outgoing_coradial_points;
                       incoming_coradial_points];

    shape = polyshape(coradial_points);
end

% Create a spiral path using some tangent as a source
% The tangent vector is
function path = spiral_path(th, K, A, B, C, f, tanget_source, stop_radius, min_points, max_points, width)
    % Initial point
    p = get_initial_point_from_second_order_curve(th, K, A, B, C, f, width);

    path = double.empty(0, 2);
    path = add_point(p, path);

    while (norm(p) < stop_radius) && (length(path) <= max_points) || (length(path) < min_points)
        T = tanget_source(p, A, B, C, f);

        % Move slightly off of the tanget vector to create a spiral
        V = T + K * perp_to_vector(T);

        STEPSIZE = 0.05;
        p = p + V * STEPSIZE;

        path = add_point(p, path);
    end
end

% Rotate a 2D vector 90 degrees counterclockwise
function P = perp_to_vector(p)
    P = [-p(2), p(1)];
end

% Calculate a coaxis vector
function M = coaxis_vector(p, A, B, C, f)
    x = p(1);
    y = p(2);

    % Equation given in the unumbered equation for M in
    % Passive Navigation as a Pattern Recognition Problem pg 4
    Mx = -A*(y^2 + f^2) + B*x*y         + C*x*f;
    My =  A*x*y         - B*(x^2 + f^2) + C*y*f;

    M = [Mx, My];
end

% Get a vector perpendicular to the coaxis vector for a given set of parameters
function Mp = perp_to_coaxis_vector(p, A, B, C, f)
    Mp = perp_to_vector(coaxis_vector(p, A, B, C, f));
end

% TODO: Get a correct starting point for a given theta on the second order curve
% E.g. the cone's intersection with the image plane starting at some
% fixed radius from where the coaxis pierces the image plane

% Need an equation for this curve first (that's not the tanget)
% generating a curve and searching the result is too inefficient.
function p = get_initial_point_from_second_order_curve(th, K, A, B, C, f, width)
    % Pick a point a fixed distance from where the coaxis pierces the image plane
    start_radius = 0.01*(1+width);
    STEPSIZE = (start_radius/1);

    p = 0.001*(1+width)*[cos(th), sin(th)] + [A*f/C, B*f/C];
end

function poly_points = add_point(p, poly_points)
    poly_points(end+1, :) = p;
end
