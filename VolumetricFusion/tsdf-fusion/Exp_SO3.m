function R = Exp_SO3(w, epsilon)
    if ~exist('epsilon','var')
        % set default value for epsilon
        epsilon = 1e-5;
    end
    S = hat_SO3(w);
    theta_2 = w'*w;
    theta = sqrt(theta_2);
    % Use Taylor expansion to avoid numerical instabilities for small theta
    % At least a check for theta != 0 would be needed.
    if theta <= epsilon
        % http://www.wolframalpha.com/input/?i=sin(theta)%2Ftheta
        A = 1 - theta_2/6;
        % https://www.wolframalpha.com/input/?i=(1+-+cos(theta))%2Ftheta%5E2
        B = 0.5 - theta_2/24;
    else
        A = sin(theta)/theta;
        B = (1 - cos(theta))/theta_2;
    end
    % Rodrigues' formula
    R = eye(3) + A*S + B*S*S;
end