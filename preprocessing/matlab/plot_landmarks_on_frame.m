function plot_landmarks_on_frame(video_path, landmarks_csv, frame_number)
%PLOT_LANDMARKS_ON_FRAME Visually verify MediaPipe landmarks and ROIs.
    landmarks = readmatrix(landmarks_csv);
    row = landmarks(frame_number, :);
    frame = read(VideoReader(video_path), frame_number);
    points = reshape(row, 2, [])';
    [rois, ~] = define_face_rois(points, [size(frame, 1) size(frame, 2)], [20 20], []);

    figure; imshow(frame); hold on;
    plot(points(:, 1), points(:, 2), 'g.', 'MarkerSize', 5);
    for roi_index = 1:numel(rois)
        box = rois(roi_index).bbox;
        rectangle('Position', [box(1) box(2) box(3)-box(1)+1 box(4)-box(2)+1], ...
            'LineWidth', 2);
        text(box(1), max(1, box(2)-4), rois(roi_index).name, 'FontWeight', 'bold');
    end
    hold off;
end
