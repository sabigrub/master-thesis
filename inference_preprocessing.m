% Definition der Dateipfade
dataDir = "./data/inference";
dataFilePath = dataDir + "/inference.txt";

% Einlesen der Daten
data = readtable(dataFilePath, 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
assert(all(sum(ismissing(data)) == 0));


function data = preprocessData(rawData)        
    % Beschleunigungsdaten extrahieren
    accX = rawData{:, 'AccX(g)'};
    accY = rawData{:, 'AccY(g)'};
    accZ = rawData{:, 'AccZ(g)'};
    
    % Berechnung der Gesamtbeschleunigung (magnitude)
    accMag = sqrt(accX.^2 + accY.^2 + accZ.^2);
    
    % Abtastfrequenz ist bekannt
    fs = 50; % Hz
    
    % Tiefpassfilter anwenden 
    cutoffFreq = 20; % Hz
    [b, a] = butter(4, cutoffFreq/(fs/2), 'low'); 
    accXFiltered = filtfilt(b, a, accX); 
    accYFiltered = filtfilt(b, a, accY); 
    accZFiltered = filtfilt(b, a, accZ); 
    accMagFiltered = filtfilt(b, a, accMag);
    
    % Hochpassfilter anwenden, um Schwerkraftkomponente zu entfernen
    cutoffFreqHigh = 0.1; % Hz
    [b, a] = butter(4, cutoffFreqHigh/(fs/2), 'high');
    accXHighpass = filtfilt(b, a, accXFiltered);
    accYHighpass = filtfilt(b, a, accYFiltered);
    accZHighpass = filtfilt(b, a, accZFiltered);
    
    % Define windowing parameters
    windowSize = 2 * fs; % 2 Sekunden
    overlap = 0.5; % 50% Überlappung
    step = round(windowSize * (1 - overlap));
    numWindows = floor((length(accXHighpass) - windowSize) / step) + 1;
    
    features = zeros(numWindows, 15);
    
    for i = 1:numWindows
        startIdx = (i-1)*step + 1;
        endIdx = startIdx + windowSize - 1;
        
        % Fenster extrahieren / Filtern
        xWindow = accXHighpass(startIdx:endIdx);
        yWindow = accYHighpass(startIdx:endIdx);
        zWindow = accZHighpass(startIdx:endIdx);
        magWindow = accMagFiltered(startIdx:endIdx);
        
        % Zeitbasierte Features
        features(i, 1:3) = [mean(xWindow), mean(yWindow), mean(zWindow)]; % Mittelwert
        features(i, 4:6) = [std(xWindow), std(yWindow), std(zWindow)]; % Standardabweichung
        features(i, 7:9) = [max(xWindow), max(yWindow), max(zWindow)]; % Maximum
        features(i, 10:12) = [min(xWindow), min(yWindow), min(zWindow)]; % Minimum
        
        % Frequenzbasierte Features
        % Remove DC offset to avoid bias
        magWindow = magWindow - mean(magWindow);
        
        % Use zero-padding to improve frequency resolution
        nfft = 512; % This gives us frequency resolution of 50/512 ≈ 0.098 Hz
        fftResult = fft(magWindow, nfft);
        fftMag = abs(fftResult);
        fftMag = fftMag(1:floor(nfft/2)+1); % Keep positive frequencies + Nyquist
        
        % Create frequency vector with improved resolution
        freqs = (0:length(fftMag)-1) * fs / nfft;
        
        % Find dominant frequency (exclude very low frequencies that aren't meaningful)
        minFreq = 0.5; % Ignore frequencies below 0.5 Hz
        maxFreq = 10;  % Focus on human movement range
        validIdx = (freqs >= minFreq) & (freqs <= maxFreq);
        
        if any(validIdx)
            validFftMag = fftMag(validIdx);
            validFreqs = freqs(validIdx);
            [~, maxIdx] = max(validFftMag);
            dominantFreq = validFreqs(maxIdx);
        else
            dominantFreq = 0; % Fallback if no valid frequency found
        end
        
        features(i, 13) = dominantFreq;
        
        % Spektrale Energie (exclude DC component at index 1)
        features(i, 14) = sum(fftMag(2:end).^2);
        
        % Debug output for first few windows
        if i <= 3
            fprintf('Window %d: Dominant frequency = %.2f Hz\n', i, dominantFreq);
        end
        
        % Dominante Frequenz
        [~, maxIdx] = max(fftMag);
        dominantFreq = freqs(maxIdx);
        features(i, 13) = dominantFreq;
    
        % Spektrale Energie
        features(i, 14) = sum(fftMag.^2);
        
        % Signal Magnitude Area (SMA)
        features(i, 15) = sum(abs(xWindow)) + sum(abs(yWindow)) + sum(abs(zWindow));    
    end
    
    % 9. Build final dataset
    featureNames = {'mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', ...
                    'max_x', 'max_y', 'max_z', 'min_x', 'min_y', 'min_z', ...
                    'dominant_freq', 'spectral_energy', 'sma'};
    data = array2table(features, "VariableNames", featureNames);
end

preprocessedData = preprocessData(data);
writetable(preprocessedData, dataDir + '/preprocessed_data.csv');
