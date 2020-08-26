% nov�rt�sim da��du k�rtu polinomu mode�us

x = dataset(:,1);
y = dataset(:,2);

n = size(y,1);
idxHO = true(n,1); %index hold-out iestatits ka true
idxValidation = randperm(n, floor(n / 3)); %randperm - iedod nejausa seciba sajauktus skaitlus lidz pirma argumenta vertibai
idxHO(idxValidation) = false;

maxDegree = 8;
% rezerv�jam vektorus mode�u nov�rt�jumiem
result = zeros(maxDegree+1,4);
resultHO = zeros(maxDegree+1,1);
resultCV = zeros(maxDegree+1,1);

% cikls, lai izveidotu visu k�rtu polinomu mode�us no 0 l�dz maxDegree
for degree = 0 : maxDegree
    %apmacibas kluda
    a = linreg(x, y, degree);
    yHat = linreg_predict(x, a, degree);
    [~, MAE, ~, ~, ~, ~, AIC, AICC, BIC] = evaluate_model(y, yHat, length(a));
    %result(degree+1,1) = MAE;
    result(degree+1,:) = [MAE AIC AICC BIC];
    
    %hold-Out kluda
    xTrain = x(idxHO,:);
    yTrain = y(idxHO);
    xValid = x(~idxHO,:); % ~ ir logiska darbiba not
    yValid = y(~idxHO);
    
    a = linreg(xTrain, yTrain, degree);
    yHat = linreg_predict(xValid, a, degree);
    [~, MAE] = evaluate_model(yValid, yHat, length(a)); % ~ nozime, ka attiecigais arguments neinterese
    resultHO(degree+1) = MAE;
    
    %LOOCV kluda
    [~, MAE] = evaluate_model_loocv(x, y, degree);
    resultCV(degree+1) = MAE;
end

% apskat�sim vizu�li, k� mode�a sare���t�ba ietekm� t� k��das nov�rt�jumu

figure;
plot(0:maxDegree, result(:,1), '-');
xlabel('Polinoma k�rta');
ylabel('MAE');

hold on;
grid on;
plot(0:maxDegree, resultHO, '-');
plot(0:maxDegree, resultCV, '-');
legend({'Apm�c�bas MAE', 'Hold-Out MAE', 'LOOCV MAE'});



figure;
plot(0:maxDegree, result(:,2), '-');
xlabel('Polinoma k�rta');
ylabel('Criterion');
hold on;
grid on;
plot(0:maxDegree, result(:,3), '-');
plot(0:maxDegree, result(:,4), '-');
legend({'AIC', 'AICC', 'BIC'});