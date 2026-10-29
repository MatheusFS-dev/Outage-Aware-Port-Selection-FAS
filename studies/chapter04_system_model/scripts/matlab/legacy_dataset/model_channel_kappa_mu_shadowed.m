function [g, J] = model_channel_kappa_mu_shadowed(W, N, kappa, m, mu)
    % Esta função gera um modelo de canal para um sistema de antena fluida baseado na distribuição kappa-mu
    % e no modelo de correlação de Jake, conforme descrito no artigo.
    
    % Gera uma distribuição kappa-mu sombreada com os parâmetros fornecidos.
    r_dist = KappaMuShadowed(kappa, m, mu, 1, N);
    r = sqrt(r_dist.multipathFading.'); % Extrai o componente de desvanecimento multipercurso

    % Gera a fase uniforme para cada caminho.
    theta = 2*pi*rand(N, 1);  % fase aprox. uniforme

    % Calcula o vetor complexo do canal para cada caminho.
    g = r .* exp(1i * theta); % N: número de portas, 1: número de transmissores

    % Inicializa a matriz de correlação cruzada entre as portas.
    J = zeros(N, N);

    % Calcula a correlação cruzada usando o modelo de Jake.
    for k = 1:N
        for l = 1:N
            % Modelo de correlação de Jake para canais entre portas.
            J(k, l) = besselj(0, (2 * pi * abs(k - l) * W) / (N - 1));
        end
    end

    % Aplica a correlação ao vetor de canais.
    [Q, L] = eigs(J, N); % Decomposição em valores singulares para calcular a raiz quadrada da matriz de correlação
    A = Q * sqrt(L); % Matriz A que estabelece a correlação entre as portas
    g = A * g; % Aplica a correlação ao vetor de canais
end
