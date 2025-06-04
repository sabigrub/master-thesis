% Define the directory containing data files
baseDir = ".";
dataDir = baseDir + "/data";
trainDataDir = dataDir + '/train';
testDataDir = dataDir + '/test';

trainDataFile = trainDataDir + '/processed/preprocessed_data.csv';
testDataFile = testDataDir + '/processed/preprocessed_data.csv';


%% 1. Dataloading
trainData = readtable(trainDataFile, 'VariableNamingRule', 'preserve');
testData = readtable(testDataFile, 'VariableNamingRule', 'preserve');

fprintf('Train Dataset Analysis:\n');
fprintf('Total train samples: %d\n', height(trainData));
fprintf('%d features and 1 targtet \n', width(trainData)-1);

% Check class imbalance
% sake of simplicity we will not handle the imbalance as its small enough
classDistribution = tabulate(trainData.Target);
classCounts = classDistribution(:, 2);
imbalanceRatio = max(classCounts) / min(classCounts);
fprintf('Imbalance ratio of train data: %.2f\n', imbalanceRatio);

%% 2. Data Shuffling
rng(42);
randomTrainIdx = randperm(height(trainData));
randomTestIdx = randperm(height(testData));
shuffledTrainData = trainData(randomTrainIdx, :);
shuffledTestData = testData(randomTestIdx, :);


% Split data in features and labels
X_train = shuffledTrainData{:, 1:end-1};  % Features
y_train = shuffledTrainData.Target;       % Labels
X_test = shuffledTestData{:, 1:end-1};    % Features
y_test = shuffledTestData.Target;         % Labels

fprintf('Training samples: %d\n', length(y_train));
fprintf('Test samples: %d\n', length(y_test));

%% 4. Train Random Forest Classifier
fprintf('Training Random Forest Classifier...\n');
numTrees = 100;
RF_model = TreeBagger(numTrees, X_train, y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on');

fprintf('Random Forest training completed.\n');

%% 5. Make Predictions on Test Set
fprintf('Making predictions on test set...\n');
RF_predictions_cell = predict(RF_model, X_test);
RF_predictions = str2double(RF_predictions_cell);

%% 6. Calculate Accuracy
RF_accuracy = sum(RF_predictions == y_test) / length(y_test) * 100;
fprintf('Random Forest Test Accuracy: %.2f%%\n', RF_accuracy);

%% 7. Display Confusion Matrix
uniqueClasses = unique(y_test);
classNames = cellstr(string(uniqueClasses));

figure;
cm_RF = confusionmat(y_test, RF_predictions);
confusionchart(cm_RF, classNames);
title('Random Forest Confusion Matrix');

%% 8. Additional Simple Metrics
fprintf('\nDetailed Results:\n');
fprintf('Correct predictions: %d out of %d\n', sum(RF_predictions == y_test), length(y_test));
fprintf('Incorrect predictions: %d\n', sum(RF_predictions ~= y_test));

% Per-class accuracy (if you want to see how each class performs)
for i = 1:length(uniqueClasses)
    classIdx = (y_test == uniqueClasses(i));
    classCorrect = sum(RF_predictions(classIdx) == y_test(classIdx));
    classTotal = sum(classIdx);
    if classTotal > 0
        classAccuracy = classCorrect / classTotal * 100;
        fprintf('Class %d accuracy: %.2f%% (%d/%d)\n', ...
            uniqueClasses(i), classAccuracy, classCorrect, classTotal);
    end
end

save(baseDir + '/classifier-rf', 'RF_model');