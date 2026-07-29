function [Ti, qi, sorted_idx, selected_quality] = select_top_k_signals(Xi, fs, k, fmin, fmax, wn)
%SELECT_TOP_K_SIGNALS Implement Equation (1) and select the top-k signals.
%
% The spectrum is the one-sided magnitude spectrum. The desired energy is
% summed around the dominant frequency inside 0.7-4 Hz; the remaining band
% energy is treated as noise. wn is measured in FFT bins.

    if nargin < 4 || isempty(fmin), fmin = 0.7; end
    if nargin < 5 || isempty(fmax), fmax = 4.0; end
    if nargin < 6 || isempty(wn), wn = 3; end

    [n_signals, signal_length] = size(Xi);
    if n_signals < k
        error('Only %d temporal signals were extracted; k=%d is required.', n_signals, k);
    end
    frequency_axis = (0:floor(signal_length / 2)) * (fs / signal_length);
    band_idx = find(frequency_axis >= fmin & frequency_axis <= fmax);
    if isempty(band_idx)
        error('No FFT bins fall inside [%.2f, %.2f] Hz.', fmin, fmax);
    end

    qi = zeros(1, n_signals);
    for signal_index = 1:n_signals
        signal = Xi(signal_index, :) - mean(Xi(signal_index, :));
        full_spectrum = abs(fft(signal));
        spectrum = full_spectrum(1:numel(frequency_axis));
        [~, local_peak] = max(spectrum(band_idx));
        peak_idx = band_idx(local_peak);
        window_idx = max(band_idx(1), peak_idx - wn):min(band_idx(end), peak_idx + wn);
        signal_energy = sum(spectrum(window_idx));
        band_energy = sum(spectrum(band_idx));
        noise_energy = max(band_energy - signal_energy, eps);
        qi(signal_index) = signal_energy / noise_energy;
    end

    [~, sorted_idx] = sort(qi, 'descend');
    selected_idx = sorted_idx(1:k);
    Ti = Xi(selected_idx, :);
    selected_quality = qi(selected_idx);
end
