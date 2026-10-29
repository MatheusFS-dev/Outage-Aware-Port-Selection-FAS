clc;
clear all;
close all;

%% ------------------- Definir Variáveis e Parâmetros de Entrada ------------------- %%
U = 1;  % Número de usuários (U = 1: FAS, U > 1: FAMA)
N = 144;  % Número de portas (subcanais) da antena fluida

% Definir SNR desejada em dB (simulando um cenário de SNR alta)
SNR_dB = 3;  % SNR em decibéis
signal_power = 1;  % Potência do sinal desejado

% Calcular a potência e o desvio padrão do ruído
noise_power = signal_power / (10^(SNR_dB / 10));  
sigma_n = sqrt(noise_power);  

% Parâmetros da distribuição kappa-mu sombreada
kappa = 0.001;  
mu = 1;  
m = 2; % Parâmetro sombreamento
W = 0.5;  % Comprimento normalizado da antena fluida (em termos de comprimento de onda)

% Potências dos usuários primário e secundário
sigma_p = 1;  
sigma_s = 1;  

% Limiar de SINR para cálculo da Outage Probability
gamma_threshold = 0.5;  % Limiar para considerar "outage"

% Outros parâmetros
N_events = 1000;  % Número de eventos para a simulação

% Inicializar matriz para armazenar coeficientes de canal + ruído
channel_plus_noise = zeros(N, U, N_events, 'like', 1i);  
SNR_events = zeros(N, N_events);

% Inicializar contador de outages de forma distribuída
outage_count = zeros(1, N_events);  

%% ---------------------- Simulação de Múltiplos Eventos em Paralelo ---------------------- %%
parpool;  % Iniciar pool de workers para paralelização

r_hat_values = zeros(U, N_events); % Preallocate an array

parfor i = 1:N_events
    fprintf('Simulando evento %d\n', i);  % Exibir progresso

    % Inicializar matriz dos coeficientes de canal
    g = zeros(N, U, 'like', 1i);  

    % Gerar coeficientes de canal kappa-mu sombreado para cada porta e usuário
    for u = 1:U
        if u == 1
            r_hat_values(u, i) = sigma_p;  % Potência do usuário primário
        else
             r_hat_values(u, i) = sigma_s;  % Potência do usuário secundário
        end
        [g(:, u), ~] = model_channel_kappa_mu_shadowed(W, N, kappa, m, mu, r_hat_values(u, i));
    end

    % Gerar ruído gaussiano aditivo branco (AWGN) para cada porta e usuário
    noise = sigma_n * (randn(N, U) + 1i * randn(N, U));  

    % Somar o ruído aos coeficientes de canal
    channel_plus_noise(:, :, i) = g + noise;

    % Calcular SNR para cada porta no evento atual
    SNR_events(:, i) = abs(g).^2 / noise_power;

    % Encontrar a melhor SNR (maior SNR entre as portas)
    max_SNR = max(SNR_events(:, i));

    % Verificar se a melhor SNR é menor que o limiar (outage)
    if max_SNR < gamma_threshold
        outage_count(i) = 1;  % Registra o "outage" no evento atual
    end
end

delete(gcp('nocreate'));  % Fechar o pool de workers

%% --------------------- Cálculo da Probabilidade de Outage --------------------- %%
outage_probability = sum(outage_count) / N_events;  

%% --------------------- Exibir Resultados --------------------- %%
fprintf('Probabilidade de Outage: %f\n', outage_probability);

%% --------------------- Salvar os Dados Gerados --------------------- %%
% Criar diretório 'runs' se não existir
if ~exist('runs', 'dir')
    mkdir('runs');
end

% Formatar nome do arquivo com base nos parâmetros
formatted_file_name = sprintf('channel_U%d_N%d_kappa%.1f_mu%.1f_m%.1f', ...
    U, N, kappa, mu, m);

% Definir pasta para salvar os resultados
generation_folder = fullfile('runs', formatted_file_name);

% Criar a pasta se não existir
if ~exist(generation_folder, 'dir')
    mkdir(generation_folder);
end

% Salvar a matriz de canal + ruído
save(fullfile(generation_folder, 'channel_plus_noise.mat'), 'channel_plus_noise');

% Salvar SNR por porta
save(fullfile(generation_folder, 'SNR_events.mat'), 'SNR_events');

% Salvar a matriz de correlação (desnormalizada)
% [~, correlation] = model_channel_kappa_mu_shadowed(W, N, kappa, m, mu, r_hat);
% correlation = 2 * correlation;  % Desnormalização
% save(fullfile(generation_folder, 'correlation.mat'), 'correlation');
