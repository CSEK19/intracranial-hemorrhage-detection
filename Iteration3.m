folder = 'Concatenated';
saveFolder = fullfile(pwd, 'Figures and Plots');
if ~isfolder(saveFolder)
    mkdir(saveFolder); % Create the folder if it does not exist
end

fileList = dir(fullfile(folder, '*.mat'));
% Identify top N components by magnitude
N = 6; % Number of components to select

combinedMatrix = [];

for k = 1:length(fileList)
    fileName = fullfile(pwd, folder, fileList(k).name);

    outputMatrix = LoadMatFile(fileName);

    combinedMatrix = [combinedMatrix; outputMatrix]; %#ok<*AGROW>
end

corrMatrix = CorrMatrix(combinedMatrix);

% Plotting the matrix scatter plots for FFT data
labels = combinedMatrix(:, end);
% Preallocate a cell array for component names
components = cell(1, N + 1);

% Build component names
for i = 1:N
    components{i} = ['Frame ', num2str(i)];
end

% Add 'Angle' at the end
components{N + 1} = 'Angle';

% Matrix Scatter Plot for All data
figureName = 'Matrix Scatter Plot for All Data';
MatrixPlotAndSave([combinedMatrix(:, 1:N), combinedMatrix(:, end - 1)], labels, 'br', 'x.', [5 5], components, figureName);

% Matrix Scatter Plot for TBI data
figureName = 'Matrix Scatter Plot for TBI Data';
idx = combinedMatrix(:, end)==1;
tbiMatrix = combinedMatrix(idx, :);
MatrixPlotAndSave([tbiMatrix(:, 1:N), tbiMatrix(:, end - 1)], [], 'r', '.', 5, components, figureName);

% Matrix Scatter Plot for Healthy data
figureName = 'Matrix Scatter Plot for Healthy Data';
idx = combinedMatrix(:, end)==0;
healthyMatrix = combinedMatrix(idx, :);
MatrixPlotAndSave([healthyMatrix(:, 1:N), healthyMatrix(:, end - 1)], [], 'b', 'x', 5, components, figureName);

% 3D Scatter Plots for All Data
Matrix3DPlotAndSave([tbiMatrix(:, 1:N), tbiMatrix(:, end - 1)], [healthyMatrix(:, 1:N), healthyMatrix(:, end - 1)], 'Frames');

% PDF Normal Distribution Plots
figureName = 'Normal Distributions';
PDFPlotAndSave(tbiMatrix(:, 1:end-2), healthyMatrix(:, 1:end-2), N, figureName);

% Apply PCA to data
[pcaMatrix, expVar] = PCAPlotAndSave(combinedMatrix, N);

% Plot Scatter Plot Matrix for PCA transformed Data
pcaLabels = pcaMatrix(:, end);
% Preallocate a cell array for component names
pcaComponents = cell(1, N + 1);

% Build component names
for i = 1:N
    pcaComponents{i} = ['PCA\_', num2str(i)];
end

% Add 'Angle' at the end
pcaComponents{N + 1} = 'Angle';

% Matrix Scatter Plot for All data
figureName = 'Matrix Scatter Plot for All PCA Data';
MatrixPlotAndSave([pcaMatrix(:, 1:N), pcaMatrix(:, end - 1)], pcaLabels, 'br', 'x.', [5 5], pcaComponents, figureName);

% Matrix Scatter Plot for TBI data
figureName = 'Matrix Scatter Plot for TBI PCA Data';
idx = pcaMatrix(:, end)==1;
tbiPCAMatrix = pcaMatrix(idx, :);
MatrixPlotAndSave([tbiPCAMatrix(:, 1:N), tbiPCAMatrix(:, end - 1)], [], 'r', '.', 5, pcaComponents, figureName);

% Matrix Scatter Plot for Healthy data
figureName = 'Matrix Scatter Plot for Healthy PCA Data';
idx = pcaMatrix(:, end)==0;
healthyPCAMatrix = pcaMatrix(idx, :);
MatrixPlotAndSave([healthyPCAMatrix(:, 1:N), healthyPCAMatrix(:, end - 1)], [], 'b', 'x', 5, pcaComponents, figureName);

% 3D Scatter Plots for All Data
Matrix3DPlotAndSave([tbiPCAMatrix(:, 1:N), tbiPCAMatrix(:, end - 1)], [healthyPCAMatrix(:, 1:N), healthyPCAMatrix(:, end - 1)], 'Principal Components');

% Create feature set with features removed through PCA
cols = [2, 7, 14, 21, 24, 29, 31, 32];
reducedMatrix = combinedMatrix(:, cols);

% Save matrices for classifier training and testing
savePath = fullfile(pwd, 'Dataset');
if ~isfolder(savePath)
    mkdir(savePath); % Create the folder if it does not exist
end
save(fullfile(savePath, 'original.mat'), 'combinedMatrix');
save(fullfile(savePath, 'reduced.mat'), 'reducedMatrix');
save(fullfile(savePath, 'transformed.mat'), 'pcaMatrix');