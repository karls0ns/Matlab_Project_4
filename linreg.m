function a = linreg(z, y, degree)
% funkcija nosaka parametrus a attiecîgâs pakâpes polinoma regresijas modelim
    m = size(z,2);
    % noslinkosim - realizçsim visu tikai viena faktora jebkuras pakâpes
    % polinoma regresijai un daudzfaktoru pirmâs pakâpes polinoma regresijai,
    % bet daudzfaktoru gadîjumam ar pakâpi >1 izvadîsim kïûdas paziòojumu
    if (degree <= 1) || (m <= 1)
        % pirmajâ kolonnâ visi vieninieki
        x = ones(size(z,1), 1);
        % pârçjâs kolonnas ir z vçrtîbas celtas pieaugoğâ pakâpç (no 1 lîdz degree)
        for i = 1 : degree
            x = [x z .^ i];
        end
        a = x \ y; % nosakâm parametrus a
    else
        error('linreg: Nav realizçts.');
    end
end
