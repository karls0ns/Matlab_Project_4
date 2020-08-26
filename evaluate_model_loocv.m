function [SAE, MAE, SSE, MSE, RMSE, R2] = evaluate_model_loocv(x, y, degree)
% funkcija aprçíina kïûdu kritçriju vçrtîbas ar LOOCV metodi
    n = size(y,1);
    diff = zeros(n,1);
    for i = 1 : n %gadijums, kura iteraciju ksaits ir vienads ar datu punktu skaitu
        %kurš punkts b?s valid?tais punkts 
        %no parejiem punktiem jaizveido apmacibas kopa un modelis
        %gatavo modeli validet izmantojot tikai 1 punktu
        
        xTrain = x;
        xTrain(i,:) = [];
        yTrain = y;          
        yTrain(i) = [];
        a = linreg(xTrain, yTrain, degree);
        xValid = x(i,:);
        yValid = y(i);
        yHat = linreg_predict(xValid, a, degree);
        diff(i) = yValid - yHat;
    end
    SAE = sum(abs(diff));
    MAE = SAE / n;
    SSE = sum(diff .^ 2);
    MSE = SSE / n;
    RMSE = sqrt(MSE);
    SSEtot = sum((y - mean(y)) .^ 2);
    R2 = 1 - SSE / SSEtot;
end
