%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% LeviantInteractive.m
%
% Display a paramterizable optical illusions inspired by Leviant
%
% Levi Burner
%
%
% History:
% 01-16-20 - Created File
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

f = figure('Units', 'normalized');
ax = axes('Parent',f,'position',[-0.075, 0.075, 0.85 0.85]);

figure_state = containers.Map;
figure_state('axis') = ax;
figure_state('illusion') = 2;

% Default parameters
figure_state('coradial_shapes') = 4;
figure_state('coradial_width') = 0.1;
figure_state('coaxis_shapes') = 8;
figure_state('coaxis_width') = 0.8;
figure_state('K') = 0.2;
figure_state('A') = 0.0;
figure_state('B') = 0.0;
figure_state('C') = 1.0;

% Wedge slider
callback = @(source, eventdata) (slider_callback(source, eventdata, figure_state, 'coradial_shapes'));
add_slider(f, callback, 0.9, 'Number of coradial shapes', 2, 20, figure_state('coradial_shapes'));

% Wedge width slider
callback = @(source, eventdata) (slider_callback(source, eventdata, figure_state, 'coradial_width'));
add_slider(f, callback, 0.8, 'Coradial shape width', 0.02, 1.0, figure_state('coradial_width'));

% Bands slider
callback = @(source, eventdata) (slider_callback(source, eventdata, figure_state, 'coaxis_shapes'));
add_slider(f, callback, 0.7, 'Number of coaxis shapes', 2, 20, figure_state('coaxis_shapes'));

% Band width slider
callback = @(source, eventdata) (slider_callback(source, eventdata, figure_state, 'coaxis_width'));
add_slider(f, callback, 0.6, 'Coaxis width', 0.02, 2.0, figure_state('coaxis_width'));

% K slider
callback = @(source, eventdata) (slider_callback(source, eventdata, figure_state, 'K'));
add_slider(f, callback, 0.5, 'K', 0.01, 1, figure_state('K'));

% A slider
range = 1.0;
callback = @(source, eventdata) (slider_callback(source, eventdata, figure_state, 'A'));
add_slider(f, callback, 0.4, 'A', -1.0, 1.0, figure_state('A'));

% B slider
callback = @(source, eventdata) (slider_callback(source, eventdata, figure_state, 'B'));
add_slider(f, callback, 0.3, 'B', -1.0, 1.0, figure_state('B'));

% C slider
callback = @(source, eventdata) (slider_callback(source, eventdata, figure_state, 'C'));
add_slider(f, callback, 0.2, 'C', -1.0, 1.0, figure_state('C'));

update(figure_state)

p = uicontrol('Parent',f,...
              'Style','popupmenu',...
              'Units','normalized',...
              'Position',[0.75, 0.05, 0.1, 0.1],...
              'String','Exponential Spiral|Coaxis Spiral',...
              'BackgroundColor', f.Color);
p.Callback = @(source, eventdata) (popup_callback(source,...
                                                  eventdata,...
                                                  figure_state));
p.Value = figure_state('illusion');

function add_slider(f, callback, y_pos, name, minimum, maximum, value)
    b = uicontrol('Parent',f,...
                  'Style','slider',...
                  'Units','normalized',...
                  'Position',[0.65,y_pos,0.3,0.05],...
                  'Min', minimum,...
                  'Max', maximum,...
                  'Value', value);
    b.Callback = callback;

    add_text(f, num2str(minimum), [0.62,y_pos,0.03,0.05])
    add_text(f, num2str(maximum), [0.95,y_pos,0.05,0.05])
    add_text(f, name, [0.70,y_pos-0.05,0.2,0.05])
end

function add_text(f, text, position)
    uicontrol('Parent',f,...
              'Style','text',...
              'Units', 'normalized',...
              'Position', position,...
              'String', text,...
              'BackgroundColor',f.Color);
end

function plot_polygons(coradial_shapes, coaxis_shapes, ax)    
    polygons = plot(ax, coradial_shapes);
    hold on
    
    for poly=polygons
       poly.FaceColor=[0, 0, 0];
       poly.FaceAlpha=1;
       poly.LineStyle='none';
    end

    polygons = plot(ax, coaxis_shapes);
    for poly=polygons
       poly.FaceColor=[0.5, 0.5, 0.5];
       poly.FaceAlpha=1;
       poly.LineStyle='none';
    end

    pbaspect([1 1 1])
    xlim([-1, 1])
    ylim([-1, 1])
    ax.Color = [1, 1, 1];
    ax.XTick=[];
    ax.YTick=[];
    ax.XColorMode='manual';
    ax.XColor=[1, 1, 1];
    ax.YColorMode='manual';
    ax.YColor=[1, 1, 1];
    hold off
end

function update(figure_state)
    display('Plot parameters')
    display(['coradial_shapes: ' num2str(figure_state('coradial_shapes'))])
    display(['coradial_width: ' num2str(figure_state('coradial_width'))])
    display(['coaxis_shapes: ' num2str(figure_state('coaxis_shapes'))])
    display(['coaxis_width: ' num2str(figure_state('coaxis_width'))])
    display(['K: ' num2str(figure_state('K'))])
    display(['A: ' num2str(figure_state('A'))])
    display(['B: ' num2str(figure_state('B'))])
    display(['C: ' num2str(figure_state('C'))])

    if figure_state('illusion') == 1
        [coradial_shapes, coaxis_shapes] = expspir(figure_state('coradial_shapes'), figure_state('coradial_width'),...
                                                   figure_state('coaxis_shapes'), figure_state('coaxis_width'),...
                                                   figure_state('K'));
    elseif figure_state('illusion') == 2
        [coradial_shapes, coaxis_shapes] = coaxis_spiral(figure_state('coradial_shapes'), figure_state('coradial_width'),...
                                                         figure_state('coaxis_shapes'), figure_state('coaxis_width'),...
                                                         figure_state('K'),...
                                                         figure_state('A'), figure_state('B'), figure_state('C'));
    end

    plot_polygons(coradial_shapes, coaxis_shapes, figure_state('axis'))
end

function slider_callback(source, eventdata, figure_state, parameter_name)
    figure_state(parameter_name) = source.Value;
    update(figure_state)
end

function popup_callback(source, eventdata, figure_state)
    figure_state('illusion') = source.Value;
    update(figure_state)
end
