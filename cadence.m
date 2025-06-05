baseDir = "./data/inference";
dataFilePath = baseDir + "/inference.txt";
data = readtable(dataFilePath, 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
assert(all(sum(ismissing(data)) == 0));

% Extracting acceleration data
accX = data{:, 'AccX(g)'};
accY = data{:, 'AccY(g)'};
accZ = data{:, 'AccZ(g)'};

% Calculate acceleration magnitude
accMag = sqrt(accX.^2 + accY.^2 + accZ.^2);

% Remove DC offset
accMag = accMag - mean(accMag);

% Parameters
fs = 50;  
windowSize = 5;  
overlap = 0.5; 

% Sliding window analysis
windowSamples = windowSize * fs;
stepSize = windowSamples * (1 - overlap);
numWindows = floor((length(accMag) - windowSamples) / stepSize) + 1;

stepFrequencies = zeros(numWindows, 1);
timeWindows = zeros(numWindows, 1);

for i = 1:numWindows
    % Window extraction
    startIdx = (i-1) * stepSize + 1;
    endIdx = startIdx + windowSamples - 1;
    window = accMag(startIdx:endIdx);
   
    % Time for this window (center point)
    timeWindows(i) = (startIdx + endIdx) / (2 * fs);
    
    % FFT for frequency analysis
    nfft = 512;
    fftResult = abs(fft(window, nfft));
    fftResult = fftResult(1:floor(nfft/2)+1);
    
    % Frequency vector
    freqs = (0:length(fftResult)-1) * fs / nfft;
    
    % Step frequency in range 1-4 Hz
    validIdx = (freqs >= 1) & (freqs <= 4);
    
    if any(validIdx)
        [~, maxIdx] = max(fftResult(validIdx));
        validFreqs = freqs(validIdx);
        stepFrequencies(i) = validFreqs(maxIdx) * 60; % Umrechnung in Schritte/Min
    else
        stepFrequencies(i) = 0;
    end
end

% Convert time to minutes
timeMinutes = timeWindows / 60;

% Plot with white background
figure('Position', [100, 100, 1000, 600], 'Color', 'w');
ax = axes('Color', 'w', 'XColor', 'k', 'YColor', 'k');

% Color coding of points 
colors = zeros(length(stepFrequencies), 3);
for i = 1:length(stepFrequencies)
    if stepFrequencies(i) < 100
        colors(i,:) = [0.2, 0.2, 0.2]; 
    elseif stepFrequencies(i) < 150
        colors(i,:) = [1, 0.3, 0.3]; 
    elseif stepFrequencies(i) < 180
        colors(i,:) = [0.3, 0.6, 1];  
    else
        colors(i,:) = [1, 0.3, 1];  
    end
end

% Scatter plot with color coding
scatter(timeMinutes, stepFrequencies, 80, colors, 'filled', 'MarkerEdgeColor', 'none');

% Axis styling
grid on;
ax.GridColor = [0.7, 0.7, 0.7];
ax.GridAlpha = 0.5;
xlabel('Zeit (Minuten)', 'Color', 'k', 'FontSize', 12);
ylabel('Schrittfrequenz (SPM)', 'Color', 'k', 'FontSize', 12);
title('Cadence', 'Color', 'k', 'FontSize', 16, 'FontWeight', 'bold');

% Y-axis limits
ylim([50, max([200, max(stepFrequencies)+20])]);
xlim([0, max(timeMinutes)]);

ax.XTickLabel = string(ax.XTick);
ax.YTickLabel = string(ax.YTick);

% Time format for X-axis (mm:ss)
xticks_minutes = 0:1:ceil(max(timeMinutes));
xticks(xticks_minutes);
xticklabels(arrayfun(@(x) sprintf('%d:%02d', floor(x), mod(x*60, 60)), xticks_minutes, 'UniformOutput', false));

% Display average value
avgStepFreq = mean(stepFrequencies(stepFrequencies > 0));
hold on;
yline(avgStepFreq, 'r--', sprintf('Ø %.0f SPM', avgStepFreq), 'LineWidth', 1.5);
hold off;

% Output detailed analysis
fprintf('\n=== step frequency analysis ===\n');
fprintf('mean step freauency: %.1f steps/minute\n', mean(stepFrequencies));
fprintf('std: %.1f SPM\n', std(stepFrequencies));
fprintf('min: %.1f SPM\n', min(stepFrequencies));
fprintf('max: %.1f SPM\n', max(stepFrequencies));
fprintf('duration: %.1f Minuten\n', max(timeMinutes));
fprintf('number of analyised windows: %d\n', numWindows);