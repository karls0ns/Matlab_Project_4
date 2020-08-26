function yHat = linreg_predict(z, a, degree)
% funkcija izmanto parametrus a, lai prognozçtu yHat vçrtîbas dotajâm z faktoru ieejas vçrtîbâm
% arguments "degree" ir polinoma pakâpe
% arguments "a" ir aprçíinâtie polinoma parametri
	m = size(z,2);
    % noslinkosim - realizçsim visu tikai viena faktora jebkuras pakâpes
    % polinoma regresijai un daudzfaktoru pirmâs pakâpes polinoma regresijai,
    % bet daudzfaktoru gadîjumam ar pakâpi >1 izvadîsim kïûdas paziòojumu
    if (degree <= 1) || (m <= 1)
        % pirmkârt jâizveido x matrica tâpat kâ linreg funkcijâ
        x = ones(size(z,1), 1);
        for i = 1 : degree
            x = [x z .^ i];
        end
        % otrkârt jâveic yHat aprçíins, izmantojot a un x
        %{
        % ğeit ir garâ versija:
        yHat = zeros(size(x,1), 1);
        for i = 1 : size(x,1)
            for j = 1 : size(x,2)
                yHat(i) = yHat(i) + a(j) * x(i,j);
            end
        end
        %}
        yHat = x * a; % îsâks pieraksts augstâk aizkomentçtajâm rindiòâm
    else
        error('linreg_predict: Nav realizçts.');
    end
end
