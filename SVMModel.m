function [accuracySVM, precisionSVM, recallSVM, f1scoreSVM] = SVMModel(XTrain, YTrain, XTest, YTest, seed)
    % GuassianModel - A static function to train and test a Support Vector 
    % Machine Classifier.
    %
    % Syntax:
    %   [accuracySVM, precisionSVM, recallSVM, f1scoreSVM] = SVMModel(XTrain, YTrain, XTest, YTest, seed);
    %
    % Input:
    %   XTrain - Features of training set.
    %   YTrain - Labels of training set.
    %   XTest - Features of testing set.
    %   YTest - Labels of testing set.
    %   seed - Random seed for SVM.
    %
    % Output:
    %   accuracySVM, precisionSVM, recallSVM, f1scoreSVM

    % Calculate class weights
    classWeights = [sum(YTrain == 0), sum(YTrain == 1)];
    costMatrix = [0, classWeights(2)/classWeights(1); classWeights(1)/classWeights(2), 0];

    rng(seed); % Set random seed for SVM
    % Train SVM model with class weights
    svmModel = fitcsvm(XTrain, YTrain, ...
        'KernelFunction', 'RBF', ...
        'KernelScale', 'auto', ...
        'BoxConstraint', 1, ...
        'Cost', costMatrix);
    
    % Predict using SVM model
    YTestPredSVM = predict(svmModel, XTest);
    
    % Evaluate the performance
    confMatSVM = confusionmat(YTest, YTestPredSVM);
    disp('Confusion Matrix for SVM:');
    disp(confMatSVM);
    
    accuracySVM = sum(diag(confMatSVM)) / sum(confMatSVM(:));
    fprintf('Accuracy of SVM: %.2f%%\n', accuracySVM * 100);
    
    % Calculate precision, recall, and F1-score for SVM
    TP_SVM = diag(confMatSVM);
    FP_SVM = sum(confMatSVM, 1)' - TP_SVM;
    FN_SVM = sum(confMatSVM, 2) - TP_SVM;
    precisionSVM = TP_SVM ./ (TP_SVM + FP_SVM);
    recallSVM = TP_SVM ./ (TP_SVM + FN_SVM);
    f1scoreSVM = 2 * (precisionSVM .* recallSVM) ./ (precisionSVM + recallSVM);
end