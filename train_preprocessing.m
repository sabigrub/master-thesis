dataDir = "./data";
trainDataDir = dataDir  + '/train';
testDataDir = dataDir + '/test';

%% Function Definition
function combinedData = processDataFiles(files, dataDir, target)
    combinedData = table();
    % Process each file
    for i = 1:length(files)
        currentFile = fullfile(dataDir, files(i).name);
        fprintf('Processing file %d of %d: %s\n', i, length(files), files(i).name);
        
        % Load the current file
        try            
            % For text files with tab delimiter
            currentData = readtable(currentFile, 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
            
            % Add a file identifier column
            currentData.FileID = repmat(i, height(currentData), 1);
            currentData.FileName = repmat(string(files(i).name), height(currentData), 1);
            currentData.Target = repmat(target, height(currentData), 1);
            
            % Check if this is the first file
            if isempty(combinedData)
                combinedData = currentData;
            else
                % Check if column names match
                if isequal(currentData.Properties.VariableNames, combinedData.Properties.VariableNames)
                    combinedData = [combinedData; currentData];
                else
                    warning('File %s has different column structure. Attempting to merge...', files(i).name);                
            
                end
            end
        catch ME
            warning('Error processing file %s: %s', files(i).name, ME.message);
        end
    end
end

function [combinedRunningData, combinedNotRunningData] = getAllData(dataDir)
    rawDataDir = dataDir + "/raw";
    notRunningDataDir = rawDataDir + "/not-running";
    runningDataDir = rawDataDir + "/running";
    
    % File pattern to match (e.g., '*.csv', '*.txt', etc.)
    filePattern = '*.txt';
    
    % Locate all matching files
    notRunningFiles = dir(fullfile(notRunningDataDir, filePattern));
    runningFiles = dir(fullfile(runningDataDir, filePattern));
    fprintf("Found %d not running data files.\n", length(notRunningFiles));
    fprintf("Found %d running data files.\n", length(runningFiles));

    combinedNotRunningData = processDataFiles(notRunningFiles, notRunningDataDir, 0);
    combinedRunningData = processDataFiles(runningFiles, runningDataDir, 1);
    fprintf('Combined running data has %d rows and %d columns.\n', height(combinedRunningData), width(combinedRunningData));
    fprintf('Combined not running data has %d rows and %d columns.\n', height(combinedNotRunningData), width(combinedNotRunningData));
    
    % Check for missing values
    assert(all(sum(ismissing(combinedNotRunningData)) == 0));
    assert(all(sum(ismissing(combinedNotRunningData)) == 0));
end

function allData = mergeAndPersist(runningData, notRunningData, dataDir)
    processedDir = dataDir + "/processed";
    notRunningDataOutput = processedDir + "/combined_not_running_data.csv";
    runningDataOutput = processedDir + "/combined_running_data.csv";
       
    % Save the combined data
    writetable(notRunningData, notRunningDataOutput);
    writetable(runningData, runningDataOutput);
    fprintf('Combined data saved to %s\n', processedDir);
    
   % Extracting acceleration data
    accDataRunning = runningData(:, {'time', 'AccX(g)', 'AccY(g)', 'AccZ(g)', 'Target'});
    accDataNotRunning = notRunningData(:, {'time', 'AccX(g)', 'AccY(g)', 'AccZ(g)', 'Target'});
    
    % Save the combined data
    writetable(accDataRunning, processedDir + '/beschleunigungsdaten_running.csv');
    writetable(accDataNotRunning, processedDir + '/beschleunigungsdaten_not_running.csv');
    
    % Read data
    allData = [accDataRunning; accDataNotRunning];
    
    % Save the combined tabel
    combinedDataOutput = processedDir+'/combined_acceleration_data.csv';
    writetable(allData, combinedDataOutput);
    fprintf('Combined data has %d rows and %d columns.\n', height(allData), width(allData));
end

function data = preprocessData(rawData)        
    % Extracting acceleration data
    accX = rawData{:, 'AccX(g)'};
    accY = rawData{:, 'AccY(g)'};
    accZ = rawData{:, 'AccZ(g)'};
    
    % Calculate acceleration magnitude
    accMag = sqrt(accX.^2 + accY.^2 + accZ.^2);
    
    % Sampling rate
    fs = 50; % Hz
    
    % Low-pass filter
    cutoffFreq = 20; % Hz
    [b, a] = butter(4, cutoffFreq/(fs/2), 'low'); accXFiltered = filtfilt(b, a, accX); accYFiltered = filtfilt(b, a, accY); accZFiltered = filtfilt(b, a, accZ); accMagFiltered = filtfilt(b, a, accMag);
    
    % High-pass filter
    cutoffFreqHigh = 0.1; % Hz
    [b, a] = butter(4, cutoffFreqHigh/(fs/2), 'high');
    accXHighpass = filtfilt(b, a, accXFiltered);
    accYHighpass = filtfilt(b, a, accYFiltered);
    accZHighpass = filtfilt(b, a, accZFiltered);
    
    % Define windowing parameters
    windowSize = 2 * fs; 
    overlap = 0.5; 
    step = round(windowSize * (1 - overlap));
    numWindows = floor((length(accXHighpass) - windowSize) / step) + 1;
 
    features = zeros(numWindows, 15);  
    
    % Initialize labels array alongside features
    windowLabels = cell(numWindows, 1);
    
    for i = 1:numWindows
        startIdx = (i-1)*step + 1;
        endIdx = startIdx + windowSize - 1;
        
        % Window extraction 
        xWindow = accXHighpass(startIdx:endIdx);
        yWindow = accYHighpass(startIdx:endIdx);
        zWindow = accZHighpass(startIdx:endIdx);
        magWindow = accMagFiltered(startIdx:endIdx);
        
        % Timebased features
        features(i, 1:3) = [mean(xWindow), mean(yWindow), mean(zWindow)];  
        features(i, 4:6) = [std(xWindow), std(yWindow), std(zWindow)];  
        features(i, 7:9) = [max(xWindow), max(yWindow), max(zWindow)];  
        features(i, 10:12) = [min(xWindow), min(yWindow), min(zWindow)]; 
       
        % Frequencybased features
        % Remove DC offset to avoid bias
        magWindow = magWindow - mean(magWindow);
        
        % FFT for frequency analysis
        nfft = 512;  
        fftResult = fft(magWindow, nfft);
        fftMag = abs(fftResult);
        fftMag = fftMag(1:floor(nfft/2)+1); 

        % Create frequency vector with improved resolution
        freqs = (0:length(fftMag)-1) * fs / nfft;
        
        % Find dominant frequency
        minFreq = 0.5;  
        maxFreq = 10;   
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
        
        % Spectral energy
        features(i, 14) = sum(fftMag(2:end).^2);
        
        % Debug output for first few windows
        if i <= 3
            fprintf('Window %d: Dominant frequency = %.2f Hz\n', i, dominantFreq);
        end
        
        % Signal Magnitude Area (SMA)
        features(i, 15) = sum(abs(xWindow)) + sum(abs(yWindow)) + sum(abs(zWindow));
    
        % Assign label based on the majority activity type in this window
        windowActivity = rawData.Target(startIdx:endIdx);
        windowLabels{i} = mode(windowActivity);
    end
    
    % 9. Build final dataset
    featureNames = {'mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', ...
                    'max_x', 'max_y', 'max_z', 'min_x', 'min_y', 'min_z', ...
                    'dominant_freq', 'spectral_energy', 'sma'};
    
    data = array2table(features, "VariableNames", featureNames);
    data.Target = cell2mat(windowLabels);
end

%% Code Execution
[trainRunningData, trainNotRunningData] = getAllData(trainDataDir);
[testRunningData, testNotRunningData] = getAllData(testDataDir);

allTrainData = mergeAndPersist(trainRunningData, trainNotRunningData, trainDataDir);
allTestData = mergeAndPersist(testRunningData, testNotRunningData, testDataDir);

trainPreprocessedData = preprocessData(allTrainData);
testPreprocessedData = preprocessData(allTestData);

writetable(trainPreprocessedData, trainDataDir + '/processed/preprocessed_data.csv');
writetable(testPreprocessedData, testDataDir + '/processed/preprocessed_data.csv');
