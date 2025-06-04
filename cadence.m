% Daten laden
baseDir = "./data/inference";
dataFilePath = baseDir + "/inference.txt";
data = readtable(dataFilePath, 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
assert(all(sum(ismissing(data)) == 0));

% Beschleunigungsdaten extrahieren
accX = data{:, 'AccX(g)'};
accY = data{:, 'AccY(g)'};
accZ = data{:, 'AccZ(g)'};

% Magnitude der Beschleunigung berechnen
accMag = sqrt(accX.^2 + accY.^2 + accZ.^2);

% DC-Offset entfernen
accMag = accMag - mean(accMag);

% Parameter
fs = 50; % Abtastrate in Hz
windowSize = 5; % Fenstergröße in Sekunden
overlap = 0.5; % 50% Überlappung

% Gleitende Fensteranalyse
windowSamples = windowSize * fs;
stepSize = windowSamples * (1 - overlap);
numWindows = floor((length(accMag) - windowSamples) / stepSize) + 1;

stepFrequencies = zeros(numWindows, 1);
timeWindows = zeros(numWindows, 1);

for i = 1:numWindows
    % Aktuelles Fenster extrahieren
    startIdx = (i-1) * stepSize + 1;
    endIdx = startIdx + windowSamples - 1;
    window = accMag(startIdx:endIdx);
    
    % Zeit für dieses Fenster (Mittelpunkt)
    timeWindows(i) = (startIdx + endIdx) / (2 * fs);
    
    % FFT für Frequenzanalyse
    nfft = 512;
    fftResult = abs(fft(window, nfft));
    fftResult = fftResult(1:floor(nfft/2)+1);
    
    % Frequenzvektor
    freqs = (0:length(fftResult)-1) * fs / nfft;
    
    % Schrittfrequenz im Bereich 1-4 Hz suchen
    validIdx = (freqs >= 1) & (freqs <= 4);
    
    if any(validIdx)
        [~, maxIdx] = max(fftResult(validIdx));
        validFreqs = freqs(validIdx);
        stepFrequencies(i) = validFreqs(maxIdx) * 60; % Umrechnung in Schritte/Min
    else
        stepFrequencies(i) = 0;
    end
end

% Zeit in Minuten umrechnen
timeMinutes = timeWindows / 60;

% Plot mit weißem Hintergrund
figure('Position', [100, 100, 1000, 600], 'Color', 'w');
ax = axes('Color', 'w', 'XColor', 'k', 'YColor', 'k');

% Farbkodierung der Punkte (ähnlich Garmin)
colors = zeros(length(stepFrequencies), 3);
for i = 1:length(stepFrequencies)
    if stepFrequencies(i) < 100
        colors(i,:) = [0.2, 0.2, 0.2]; % Grau für niedrige Werte
    elseif stepFrequencies(i) < 150
        colors(i,:) = [1, 0.3, 0.3]; % Rot für mittlere Werte
    elseif stepFrequencies(i) < 180
        colors(i,:) = [0.3, 0.6, 1]; % Blau für hohe Werte
    else
        colors(i,:) = [1, 0.3, 1]; % Magenta für sehr hohe Werte
    end
end

% Scatter Plot mit Farbkodierung
scatter(timeMinutes, stepFrequencies, 80, colors, 'filled', 'MarkerEdgeColor', 'none');

% Achsen-Styling
grid on;
ax.GridColor = [0.7, 0.7, 0.7];
ax.GridAlpha = 0.5;
xlabel('Zeit (Minuten)', 'Color', 'k', 'FontSize', 12);
ylabel('Schrittfrequenz (SPM)', 'Color', 'k', 'FontSize', 12);
title('Cadence', 'Color', 'k', 'FontSize', 16, 'FontWeight', 'bold');

% Y-Achse Limits
ylim([50, max([200, max(stepFrequencies)+20])]);
xlim([0, max(timeMinutes)]);

% Schwarze Tick-Labels
ax.XTickLabel = string(ax.XTick);
ax.YTickLabel = string(ax.YTick);

% Zeitformat für X-Achse (mm:ss)
xticks_minutes = 0:1:ceil(max(timeMinutes));
xticks(xticks_minutes);
xticklabels(arrayfun(@(x) sprintf('%d:%02d', floor(x), mod(x*60, 60)), xticks_minutes, 'UniformOutput', false));
% Durchschnittswert anzeigen
avgStepFreq = mean(stepFrequencies(stepFrequencies > 0));
hold on;
yline(avgStepFreq, 'r--', sprintf('Ø %.0f SPM', avgStepFreq), 'LineWidth', 1.5);
hold off;

% Detaillierte Analyse ausgeben
fprintf('\n=== Schrittfrequenz-Analyse ===\n');
fprintf('Durchschnittliche Schrittfrequenz: %.1f Schritte/Minute\n', mean(stepFrequencies));
fprintf('Standardabweichung: %.1f SPM\n', std(stepFrequencies));
fprintf('Minimum: %.1f SPM\n', min(stepFrequencies));
fprintf('Maximum: %.1f SPM\n', max(stepFrequencies));
fprintf('Gesamtdauer: %.1f Minuten\n', max(timeMinutes));
fprintf('Anzahl analysierte Fenster: %d\n', numWindows);