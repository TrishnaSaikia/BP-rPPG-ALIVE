function Xi = extract_block_temporal_signals(video_path, landmarks, start_frame, num_frames, block_size)
%EXTRACT_BLOCK_TEMPORAL_SIGNALS Extract green-channel signals for one clip.
%
% The ROIs follow MediaPipe landmarks in every frame. A fixed grid count is
% established from the first valid frame in the clip, ensuring that each
% block has a consistent temporal identity while its pixel location follows
% facial movement.

    if nargin < 5 || isempty(block_size), block_size = [20 20]; end
    end_frame = start_frame + num_frames - 1;
    if end_frame > size(landmarks, 1)
        error('Requested clip exceeds available landmark rows.');
    end

    first_valid = [];
    for frame_number = start_frame:end_frame
        row = landmarks(frame_number, :);
        if any(isfinite(row))
            first_valid = frame_number;
            break;
        end
    end
    if isempty(first_valid)
        error('No face landmarks were detected in frames %d-%d.', start_frame, end_frame);
    end

    reader = VideoReader(video_path);
    first_frame = read(reader, first_valid);
    frame_size = [size(first_frame, 1) size(first_frame, 2)];
    first_landmarks = reshape(landmarks(first_valid, :), 2, [])';
    [last_rois, grid_shapes] = define_face_rois(first_landmarks, frame_size, block_size, []);
    total_blocks = sum(arrayfun(@(roi) size(roi.blocks, 1), last_rois));
    Xi = zeros(total_blocks, num_frames);

    for frame_number = start_frame:end_frame
        frame = read(reader, frame_number);
        row = landmarks(frame_number, :);
        if any(isfinite(row))
            current_landmarks = reshape(row, 2, [])';
            try
                [current_rois, ~] = define_face_rois(current_landmarks, frame_size, block_size, grid_shapes);
                last_rois = current_rois;
            catch
                % Retain the most recent valid ROIs for an isolated failed frame.
            end
        end

        green = double(frame(:, :, 2));
        column = frame_number - start_frame + 1;
        block_index = 1;
        for roi_index = 1:numel(last_rois)
            blocks = last_rois(roi_index).blocks;
            for local_index = 1:size(blocks, 1)
                box = blocks(local_index, :);
                pixels = green(box(2):box(4), box(1):box(3));
                Xi(block_index, column) = mean(pixels(:));
                block_index = block_index + 1;
            end
        end
    end
end
