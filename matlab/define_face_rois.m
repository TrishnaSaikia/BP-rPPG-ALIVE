function [rois, grid_shapes] = define_face_rois(landmarks, frame_size, block_size, fixed_grid_shapes)
%DEFINE_FACE_ROIS Define ROIs and a non-overlapping block grid.
%
% We use forehead, cheek and chin facial regions while
% excluding eye and mouth areas. Each ROI is represented by the minimum
% enclosing rectangle of its MediaPipe landmarks. The rectangle is divided
% into non-overlapping blocks.
%
% landmarks         : 468 x 2 MediaPipe pixel coordinates for one frame
% frame_size         : [height width]
% block_size         : desired [height width] used to determine the grid on
%                      the first valid frame (default [20 20])
% fixed_grid_shapes  : optional R x 2 [rows cols]. When supplied, the same
%                      grid count is retained while the ROI follows landmarks
%                      in later frames.
%
% rois fields: name, bbox [x1 y1 x2 y2], blocks [x1 y1 x2 y2]

    if nargin < 3 || isempty(block_size), block_size = [20 20]; end
    if nargin < 4, fixed_grid_shapes = []; end
    if size(landmarks, 1) ~= 468 || size(landmarks, 2) ~= 2
        error('landmarks must have size 468 x 2.');
    end

    % MediaPipe indices are zero-based here and converted below for MATLAB.
    idx.forehead    = [10 109 67 103 54 21 251 284 332 297 338 9 8];
    idx.left_cheek  = [50 101 118 119 120 121 128 142 203 205 206 207];
    idx.right_cheek = [280 330 347 348 349 350 357 371 423 425 426 427];
    idx.chin        = [152 175 199 200 201 208 421 428];

    roi_names = {'Forehead', 'LeftCheek', 'RightCheek', 'Chin'};
    fields = {'forehead', 'left_cheek', 'right_cheek', 'chin'};
    n_rois = numel(roi_names);
    rois = repmat(struct('name', '', 'bbox', [], 'blocks', []), n_rois, 1);
    grid_shapes = zeros(n_rois, 2);

    for r = 1:n_rois
        points = landmarks(idx.(fields{r}) + 1, :);
        points = points(all(isfinite(points), 2), :);
        if isempty(points)
            error('No valid landmarks available for ROI %s.', roi_names{r});
        end

        x1 = max(1, floor(min(points(:, 1))));
        x2 = min(frame_size(2), ceil(max(points(:, 1))));
        y1 = max(1, floor(min(points(:, 2))));
        y2 = min(frame_size(1), ceil(max(points(:, 2))));
        if x2 < x1 || y2 < y1
            error('Invalid bounding box for ROI %s.', roi_names{r});
        end

        if isempty(fixed_grid_shapes)
            n_rows = max(1, floor((y2 - y1 + 1) / block_size(1)));
            n_cols = max(1, floor((x2 - x1 + 1) / block_size(2)));
        else
            n_rows = fixed_grid_shapes(r, 1);
            n_cols = fixed_grid_shapes(r, 2);
        end
        grid_shapes(r, :) = [n_rows n_cols];
        bbox = [x1 y1 x2 y2];
        rois(r).name = roi_names{r};
        rois(r).bbox = bbox;
        rois(r).blocks = split_bbox(bbox, n_rows, n_cols);
    end
end

function blocks = split_bbox(bbox, n_rows, n_cols)
% Use pixel-edge coordinates so neighbouring blocks never share pixels.
    x_edges = round(linspace(bbox(1), bbox(3) + 1, n_cols + 1));
    y_edges = round(linspace(bbox(2), bbox(4) + 1, n_rows + 1));
    blocks = zeros(n_rows * n_cols, 4);
    counter = 1;
    for row = 1:n_rows
        for col = 1:n_cols
            x1 = x_edges(col);
            x2 = max(x1, x_edges(col + 1) - 1);
            y1 = y_edges(row);
            y2 = max(y1, y_edges(row + 1) - 1);
            blocks(counter, :) = [x1 y1 x2 y2];
            counter = counter + 1;
        end
    end
end
