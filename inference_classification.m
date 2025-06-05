% Define the directory containing data files
baseDir = ".";
dataFilePath = baseDir + "/data/inference/preprocessed_data";
modelPath = baseDir + '/classifier-rf';

data = readtable(dataFilePath, 'VariableNamingRule', 'preserve');
model = load(modelPath).RF_model;
X_inference = data{:, :};
predictions = predict(model, X_inference);

% Convert cell array of strings to numeric if needed
if iscell(predictions)
    predictions_numeric = str2double(predictions);
else
    predictions_numeric = predictions;
end

% Count the predictions
count_0 = sum(predictions_numeric == 0);
count_1 = sum(predictions_numeric == 1);
total_predictions = length(predictions_numeric);

% Display the counts
fprintf('Prediction counts:\n');
fprintf('Class 0 (not running): %d predictions (%.1f%%)\n', count_0, (count_0/total_predictions)*100);
fprintf('Class 1 (running): %d predictions (%.1f%%)\n', count_1, (count_1/total_predictions)*100);
fprintf('Total predictions: %d\n', total_predictions);

fprintf('\nDetailed breakdown:\n');
tabulate(predictions_numeric)

writetable(array2table(predictions), baseDir + '/data/inference/results.csv')