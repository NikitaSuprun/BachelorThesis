% Read in data
input = readtable('./Data_100k/LHS_parameters_m.txt'); % Input data 
% parameters (31) (unseen)
output = readtable('./Data_100k/CellPerformance.txt'); % Output data 
% parameters (4) (unseen)

% Data processing
X = [log_normalise(input.LH), log_normalise(input.LP), ...
    log_normalise(input.LE), log_normalise(input.muHh), ...
    log_normalise(input.muPh), log_normalise(input.muPe), ...
    log_normalise(input.muEe), log_normalise(input.NvH), ...
    log_normalise(input.NcH), log_normalise(input.NvE), ...
    log_normalise(input.NcE), log_normalise(input.NvP), ...
    log_normalise(input.NcP), log_normalise(input.chiHh), ...
    log_normalise(input.chiHe), log_normalise(input.chiPh), ...
    log_normalise(input.chiPe), log_normalise(input.chiEh), ...
    log_normalise(input.chiEe), log_normalise(input.Wlm), ...
    log_normalise(input.Whm), log_normalise(input.epsH), ...
    log_normalise(input.epsP), log_normalise(input.epsE), ...
    log_normalise(input.Gavg), log_normalise(input.Aug), ...
    log_normalise(input.Brad), log_normalise(input.Taue), ...
    log_normalise(input.Tauh), log_normalise(input.vII), ...
    log_normalise(input.vIII)];
    
Y = output.PCE;

% Split processed data into training and test set
% 80% training set, 0% validation set and 20% test set.
[train_idx, ~, test_idx] = dividerand(size(input,1), 0.8, 0, 0.2);
    
X_train = X(train_idx, :);
Y_train = Y(train_idx, :);
    
X_test = X(test_idx, :);
Y_test = Y(test_idx, :);

% Train NN
PCE_NN = fitrnet(X_train,Y_train,"OptimizeHyperparameters","auto", ...
    "HyperparameterOptimizationOptions",struct("AcquisitionFunctionName", ...
    "expected-improvement-plus"));

% Save NN
outputDir = "NN";
outputFile = fullfile(outputDir, "PCE_NN.mat");
save(outputFile, "PCE_NN");

% Predict PCE for test set
Y_predicted = predict(PCE_NN, X_test);

% Evaluate the model
Y_mse = mean((Y_predicted - Y_test).^2); % Mean Squared Error
Y_rmse = sqrt(Y_mse);
Y_r2 = 1 - sum((Y_predicted - Y_test).^2)/sum((Y_test - mean(Y_test)).^2);

disp("NN performance")
disp(['Mean Squared Error: ' num2str(Y_mse)]);
disp(['Root Mean Squared Error: ' num2str(Y_rmse)]);
disp(['R^2: ' num2str(Y_r2)]);

% Plot NN performance
figure();
scatter(Y_test, Y_predicted, 'filled', 'SizeData', 4);
hold on;

% Plot a diagonal line for reference (perfect prediction)
plot(Y_test, Y_test, 'r--','LineWidth', 1);
xlim tight;

xlabel('Actual Values ');
ylabel('Predicted Values');

function normalised = log_normalise(x)
    normalised = log(x)/mean(log(x));   
end