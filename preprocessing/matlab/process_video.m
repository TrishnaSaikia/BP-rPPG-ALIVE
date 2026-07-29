function process_video(video_path, landmarks_csv, output_dir, dataset_name, subject_id, video_id, quality_name, sbp, dbp, block_size)
%PROCESS_VIDEO Prepare all non-overlapping 4-second rPPG clips from a video.
%
% Outputs per subject:
%   rppg_topk.csv    - one row per clip, K x L values flattened row-wise
%   clip_quality.csv - mean quality of the selected top-k signals
%   labels.csv       - SBP, DBP repeated for every clip
%   metadata.csv     - keys used to match synchronized PPG clips
%
% The final incomplete clip is discarded, as the network requires fixed
% four-second inputs.

    if nargin < 10 || isempty(block_size), block_size = [20 20]; end
    clip_seconds = 4;
    k = 15;
    expected_fps = 30;

    landmarks = readmatrix(landmarks_csv);
    if size(landmarks, 2) ~= 936
        error('Landmark CSV must contain 936 numeric columns without a header.');
    end
    reader = VideoReader(video_path);
    if abs(reader.FrameRate - expected_fps) > 0.1
        error('The paper configuration expects 30 fps video; found %.4f fps.', reader.FrameRate);
    end
    frames_per_clip = clip_seconds * expected_fps;
    n_clips = floor(size(landmarks, 1) / frames_per_clip);
    if n_clips < 1
        error('Video does not contain one complete four-second clip.');
    end

    rppg_rows = zeros(n_clips, k * frames_per_clip);
    quality_rows = zeros(n_clips, 1);
    labels = repmat([sbp dbp], n_clips, 1);
    clip_indices = (0:n_clips-1)';
    start_times = clip_indices * clip_seconds;

    for clip = 1:n_clips
        start_frame = (clip - 1) * frames_per_clip + 1;
        Xi = extract_block_temporal_signals(video_path, landmarks, start_frame, frames_per_clip, block_size);
        [Ti, ~, ~, selected_quality] = select_top_k_signals(Xi, expected_fps, k, 0.7, 4.0, 3);
        % Transpose before reshape so PyTorch view(K,L) restores signal rows.
        rppg_rows(clip, :) = reshape(Ti.', 1, []);
        quality_rows(clip) = mean(selected_quality);
    end

    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    writematrix(rppg_rows, fullfile(output_dir, 'rppg_topk.csv'));
    writematrix(quality_rows, fullfile(output_dir, 'clip_quality.csv'));
    writematrix(labels, fullfile(output_dir, 'labels.csv'));
    metadata = table(
        repmat(string(dataset_name), n_clips, 1), ...
        repmat(string(subject_id), n_clips, 1), ...
        repmat(string(video_id), n_clips, 1), ...
        repmat(string(quality_name), n_clips, 1), ...
        clip_indices, start_times, ...
        'VariableNames', {'dataset','subject_id','video_id','quality','clip_index','start_time_sec'});
    writetable(metadata, fullfile(output_dir, 'metadata.csv'));
end
