function [] = function_main_SINR_onlyPorts_kappaMu(kappa, mu, m, W, N, U, N_events)

%% ------------------- Definir Variáveis e Parâmetros de Entrada ------------------- %%
saveData = 1;

% Potências dos usuários primário e secundário
sigma_p = 1;
sigma_s = 1;

%% ---------------------- Simulação de Múltiplos Eventos ---------------------- %%

SNR_events = zeros(N_events, N);

parfor i = 1:N_events

    if(mod(i,100) == 0)
        fprintf('Simulando evento %d\n', i);  % Exibir progresso
    end

    % Inicializar matriz dos coeficientes de canal
    g = zeros(N, U);

    % Gerar coeficientes de canal kappa-mu sombreado para cada porta e usuário
    for u = 1:U
        if u == 1
            r_hat = sigma_p;  % Potência do usuário primário
        else
            r_hat = sigma_s;  % Potência do usuário secundário
        end
        [g(:, u), ~] = model_channel_kappa_mu_shadowed(W, N, kappa, m, mu);
    end

    % Calcular SNR para cada porta no evento atual
    SNR_events(i, :) = abs(g).^2;
end

figure(1);
plot(SNR_events(1,:))
hold on
for i=2:N
    plot(SNR_events(i,:))
end
set(gca, 'Yscale', 'log');

%% --------------------- Salvar os Dados Gerados --------------------- %%
if(saveData==1)
    % Criar diretório 'runs' se não existir
    if ~exist('runs', 'dir')
        mkdir('runs');
    end

    % Formatar nome do arquivo com base nos parâmetros
    formatted_file_name = sprintf('channel_W%1.1f_U%d_N%d_kappa%1.1e_mu%.1f_m%.1f', ...
        W, U, N, kappa, mu, m);

    % Definir pasta para salvar os resultados
    generation_folder = fullfile('runs', formatted_file_name);

    % Criar a pasta se não existir
    if ~exist(generation_folder, 'dir')
        mkdir(generation_folder);
    end

    SNR_file = sprintf('SNR_events_W%1.1f_U%d_N%d_kappa%1.1e_mu%.1f_m%.1f.mat', ...
        W, U, N, kappa, mu, m);

    % Salvar SNR por porta
    save(fullfile(generation_folder, SNR_file), 'SNR_events');

    SNR_figure = sprintf('SNR_events_W%1.1f_U%d_N%d_kappa%1.1e_mu%.1f_m%.1f.pdf', ...
        W, U, N, kappa, mu, m);
    saveas(gcf, fullfile(generation_folder, SNR_figure));
end