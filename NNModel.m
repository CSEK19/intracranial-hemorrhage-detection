function [accuracyNN, precisionNN, recallNN, f1scoreNN] = NNModel(XTrain, YTrain, XTest, YTest, seed)
    % GuassianModel - A static function to train and test a Neural Network 
    % Classifier.
    %
    % Syntax:
    %   [accuracyNN, precisionNN, recallNN, f1scoreNN] = NNModel(XTrain, YTrain, XTest, YTest, seed);
    %
    % Input:
    %   XTrain - Features of training set.
    %   YTrain - Labels of training set.
    %   XTest - Features of testing set.
    %   YTest - Labels of testing set.
    %   seed - Random seed for SVM.
    %
    % Output:
    %   accuracyNN, precisionNN, recallNN, f1scoreNN

    % Train Neural Network model
    rng(seed); % Set random seed for NN
    neuralNetModel = fitcnet(XTrain, YTrain);
    
    % Predict using Neural Network model
    YTestPredNN = predict(neuralNetModel, XTest);
    
    % Evaluate the performance
    confMatNN = confusionmat(YTest, YTestPredNN);
    disp('Confusion Matrix for Neural Network:');
    disp(confMatNN);
    
    accuracyNN = sum(diag(confMatNN)) / sum(confMatNN(:));
    fprintf('Accuracy of Neural Network: %.2f%%\n', accuracyNN * 100);
    
    % Calculate precision, recall, and F1-score for Neural Network model
    TP_NN = diag(confMatNN);
    FP_NN = sum(confMatNN, 1)' - TP_NN;
    FN_NN = sum(confMatNN, 2) - TP_NN;
    precisionNN = TP_NN ./ (TP_NN + FP_NN);
    recallNN = TP_NN ./ (TP_NN + FN_NN);
    f1scoreNN = 2 * (precisionNN .* recallNN) ./ (precisionNN + recallNN);
end