function [ predict_x, predict_y, state, param ] = kalmanFilter( t, x, y, state, param, previous_t )

% Takes a set of sensor measurements and previous state to return
% the max a-posteriori state and uncertainity of a **2D linear dynamical
% system**

% Argument params:  
% x and y are sensor measurements (z(t))
% state = [x(t-1),y(t-1),xdot(t-1),ydot(t-1)]
% param carrying last state uncertainity matrix P (in param.P)

% Returns
% state = [x(t),y(t),xdot(t),ydot(t)]
% predict_x / predict_y: redundant as x(t) and y(t)
% param: carries updated state uncertainity

    %% Parameters:
    % Sensor rate
    dt = 0.033;
    
    % Uncertaininties
    sen_noise = 10e-2;
    state_noise = 10e-3;
    
    % Linear state transition
    % x(t) = A*x(t-1) where x(t) is [x, xdot]
    % State transition matrix A (decoupled system)
    Ax = [1 dt;0 1];
    Ay = Ax;
    
    % Simple measurement uncetainity
    % z(t) = C*x(t) + noise
    % where noise = eye(2) * sen_noise assuming all readings same noise
    C = eye(2);  
    % Sensor noise (diagonal covariance - constant)
    R = eye(2)*sen_noise;

    % Check if the first time running this function
    if previous_t<0
        state = [x, y, 0, 0];
        param.P = state_noise * eye(4);
        Px = param.P(1:2,1:2);
        Py = param.P(3:4,3:4);
        predict_x = x;
        predict_y = y;
        return;
    end
    
    % We now have z(t) as fn(x(t)) and x(t) as fn(x(t-1))
    % p((x(t)|x(t-1)) = gauss(A*x(t-1), A*P(t-1)*A' + State_Noise)
    % p((z(t)|x(t)) = gauss(C*x(t), C*P(t)*C' + Sen_Noise)
    % Where P is the state uncertainity
    
    %% Kalman filter updates
    
    % Grouping uncertainities from previous state noise and state transition 
    Px = Ax*param.P(1:2,1:2)*Ax'+eye(2)*state_noise;
    Py = Ay*param.P(3:4,3:4)*Ay'+eye(2)*state_noise;
    
    % Kalman gain K as a function of R and P - the measurement and state uncertainities
    Kx = Px*C'/(R+C*Px*C');
    Ky = Py*C'/(R+C*Py*C');
       
    % Max a-posteriori estimates of x(t+1) depending on state x(t) and measurement
    pr_x = Ax*[state(1),state(3)]'+Kx*([x,0]'-C*Ax*[state(1),state(3)]');
    pr_y = Ay*[state(2),state(4)]'+Ky*([y,0]'-C*Ay*[state(2),state(4)]');
    
    predict_x = pr_x(1);
    predict_y = pr_y(1);
    
    state = [pr_x(1), pr_y(1), pr_x(2), pr_y(2)];
    
    % Max a-posteriori state uncertainity update
    new_Px = Px - Kx*C*Px;
    new_Py = Py - Ky*C*Py;
    
    param.P = [new_Px, eye(2);eye(2),new_Py];
end
