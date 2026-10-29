clc
clear all

%% Parâmetros de Entrada
U = 1; % Número de usuários, U = 1: FAS (Full Antenna Selection), U > 1: FAMA (Full Antenna Multiple Access)
N = 144; % Número de portas (subcanais) da antena fluida
%% Definir a SNR desejada em dB
SNR_dB = 30; % 10 dB
% Potência do sinal desejado (usuário primário)
signal_power = 1; % Assumindo que a potência do sinal seja 1 (ou outro valor apropriado)
 
% Calcular a potência do ruído a partir da SNR
noise_power = signal_power / (10^(SNR_dB / 10));
 
% Calcular o desvio padrão do ruído (sigma_n) (a potência do ruído é sigma_n^2)
sigma_n = sqrt(noise_power);

kappa = 1; % Parâmetro de distribuição kappa-mu (parâmetro de propagação)
mu = 2; % Parâmetro de distribuição kappa-mu (número de múltiplos caminhos)
m = 3; % Parâmetro de sombra na distribuição kappa-mu sombreada
W = 0.5; % Comprimento normalizado da antena fluida (em termos de comprimentos de onda)
sigma_p = 1; % Potência do usuário primário
sigma_s = 1; % Potência do usuário secundário
N_events = 100; % Número de eventos para simulação

% Definir o limiar de SINR para o cálculo da probabilidade de "outage"
gamma_threshold = 0.7; % Limiar de SINR (razão sinal-ruído + interferência)
outage_count = 0; % Inicializa a contagem de instâncias de "outage"

gamma_k = zeros(N_events, N); % Vetor para armazenar a SINR de cada porta
% Loop para simulação de múltiplos eventos
for i = 1:N_events
    %% Printando qual evento é o atual
    fprintf('Event %d\n', i);

    %% Inicialização dos coeficientes de canal complexos
    g = zeros(N, U); % Matriz para armazenar os coeficientes de canal para cada usuário e porta

    %% Gerar coeficientes de canal kappa-mu sombreado para cada usuário e porta
    % Gera os coeficientes de canal chamando a função model_channel_kappa_mu_shadowed
    for u = 1:U
        if u == 1
            r_hat = sigma_p; % Potência do sinal desejado do usuário primário
        else
            r_hat = sigma_s; % Potência do sinal do usuário secundário
        end
        [g(:, u), ~] = model_channel_kappa_mu_shadowed(W, N, kappa, m, mu, r_hat);
    end

    %% Inicialização de variáveis para armazenar os valores ótimos
    gamma_max = -Inf; % Inicializa o valor máximo de SINR como -Inf
    k_opt = 1; % Índice da porta que maximiza a SINR

    %% Calcular a SINR para cada porta e encontrar a máxima
    
    for k = 1:N
        % Potência do sinal desejado (usuário primário)
        signal_power = abs(g(k, 1))^2;

        % Potência de interferência (usuários secundários) e ruído
        interference_power = sum(abs(g(k, 2:U)).^2); % Soma das potências de interferência
        noise_power = sigma_n^2; % Potência do ruído

        % Cálculo da SINR para a porta 'k'
        gamma_k(i,k) = signal_power / (interference_power + noise_power);

        % Verifica se a SINR atual é a maior encontrada até agora
        if gamma_k(i,k) > gamma_max
            gamma_max = gamma_k(i,k); % Atualiza o valor máximo de SINR
            k_opt = k; % Atualiza o índice da porta ótima
        end
    end

    %% Cálculo da Outage Probability
    % Verifica se a SINR máxima está abaixo do limiar definido
    if gamma_max < gamma_threshold
        outage_count = outage_count + 1; % Incrementa a contagem de "outage"
    end
end

% Cálculo da Outage Probability
outage_probability = outage_count / N_events; % Probabilidade de "outage" baseada na simulação

%% Exibir Resultados da Outage Probability
fprintf('Probabilidade de Outage: %f\n', outage_probability);

%% Exibir Resultados
% Exibe a porta ótima e o valor máximo da SINR
fprintf('Porta ótima para maximizar a SINR: %d\n', k_opt);
fprintf('Valor máximo da SINR: %f\n', gamma_max);

%% ---------------------------- Save the Results ---------------------------- %%
% Create the 'runs' directory if it doesn't exist
if ~exist('runs', 'dir')
    mkdir('runs');
end

% Formatação do nome do arquivo com base nos parâmetros
formatted_file_name = sprintf('generation_U%d_N%d_kappa%.1f_mu%.1f_m%.1f', U, N, kappa, mu, m);

% Define a common folder for all generations
generation_folder = fullfile('runs', formatted_file_name);

% Create the folder if it doesn't exist
if ~exist(generation_folder, 'dir')
    mkdir(generation_folder);
end

% Save the results into files in the common folder
save(fullfile(generation_folder, 'SINR.mat'), 'gamma_k');
save(fullfile(generation_folder, 'outage_probability.mat'), 'outage_probability');
