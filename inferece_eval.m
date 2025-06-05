baseDir = "./data/inference";
dataFilePath = baseDir + "/results.csv";

predictions = readtable(dataFilePath, 'VariableNamingRule', 'preserve').predictions;

% Create time vector 
time_minutes = (1:length(predictions)) / 60; % 1 window is 2 seconds but with 50% overalp

figure('Position', [100, 100, 1200, 600]);

% Create the main plot
subplot(2,1,1);
area(time_minutes, predictions, 'FaceColor', [0.7, 0.8, 1], 'EdgeColor', [0.2, 0.4, 0.8], 'LineWidth', 2);
hold on;
% Add scatter points for running periods
running_idx = find(predictions == 1);
if ~isempty(running_idx)
    scatter(time_minutes(running_idx), predictions(running_idx), 50, 'b', 'filled', 'MarkerFaceColor', [0.2, 0.4, 0.8]);
end
title('Aktivitätsklassifizierung - Laufen', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.2, 0.4, 0.8]);
xlabel('Zeit (min)', 'FontSize', 12);
ylabel('Aktivitätslevel', 'FontSize', 12); 
ylim([-0.1, 1.1]);
grid on;
set(gca, 'YTick', [0, 1], 'YTickLabel', {'Nicht-Laufen', 'Laufen'});
set(gca, 'Color', [0.95, 0.98, 1]);

% Create a summary subplot
subplot(2,1,2);
total_time = length(predictions);
active_time = sum(predictions);
rest_time = total_time - active_time;

% Pie chart
pie_data = [active_time, rest_time];
pie_labels = {sprintf('Laufen \n%.1f%%', (active_time/total_time)*100), ...
              sprintf('Nicht Laufen \n%.1f%%', (rest_time/total_time)*100)};

pie(pie_data, pie_labels);
colormap([0.7, 0.8, 1; 0.9, 0.95, 1]);
title('Zusammenfassung der Klassifikation', 'FontSize', 14, 'FontWeight', 'bold');

% Save the plot
saveas(gcf, baseDir + 'activity_plot.png');
fprintf('Plot saved as activity_plot.png!');