%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Script for Correlated kappa-mu Shadowed FAS Model
%  Generates dataset and saves it to a .mat file with parameters in filename
%
%  Author : Pedro Marcio Raposo Pereira
%  Year   : 2025
%
%  License: MIT License
%           Permission is hereby granted, free of charge, to any person
%           obtaining a copy of this software and associated documentation
%           files (the "Software"), to deal in the Software without
%           restriction, including without limitation the rights to use,
%           copy, modify, merge, publish, distribute, sublicense, and/or
%           sell copies of the Software, and to permit persons to whom the
%           Software is furnished to do so, subject to the following
%           conditions:
%
%           The above copyright notice and this permission notice shall be
%           included in all copies or substantial portions of the Software.
%
%           THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
%           EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
%           OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
%           NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
%           HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
%           WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
%           FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
%           OTHER DEALINGS IN THE SOFTWARE.
%
%  References:
%   [1] INCLUDE OUR REFERENCE AFTER PUBLICATION 
%   [2] F. J. Lopez-Martinez, J. F. Paris, and J. M. Romero-Jerez,
%       "The kappa-mu Shadowed Fading Model: Unifying the kappa-mu and
%       eta-mu Distributions," IEEE Trans. Veh. Technol., vol. 65, no. 12,
%       pp. 9630–9641, Dec. 2016.
%
%   [3] W. Waqar, M. Shafiq, S. Salehi, M. Egan, and J. Thompson,
%       "Deep Learning Enabled Slow Fluid Antenna Multiple Access,"
%       IEEE Commun. Lett., vol. 27, no. 3, pp. 861–865, 2023.'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all; hold off;

%% Simulation and Channel Parameters
kappa       = 5;          % Ratio of dominant to scattered power
mu          = 2;          % Number of multipath clusters
m           = 1;          % Shadowing severity parameter (m -> inf is no shadowing)
r_hat       = 1;          % Target average power
N           = 100;        % Number of ports in the fluid antenna
W           = 1;          % Normalized size of the antenna (in wavelengths)
NumSamples  = 1e6;        % Number of Monte Carlo samples

%% Auxiliary Variables (Derived from parameters)
total_dominant_power = (r_hat * kappa) / (1 + kappa);
total_scatter_power  = r_hat / (1 + kappa);

dominant_power_per_cluster = total_dominant_power / mu;
p_i = sqrt(dominant_power_per_cluster / 2);
q_i = p_i;
rho_i = p_i + 1j * q_i; % Complex dominant amplitude per cluster

scatter_power_per_cluster = total_scatter_power / mu;

%% FAS Correlation Matrix Generation
correlation = zeros(N, N);
for k = 1:N
    for l = 1:N
        correlation(k, l) = besselj(0, 2 * pi * abs(k - l) * W / (N - 1));
    end
end
correlation = (correlation + correlation') / 2; % due to numerical precision errors
[U, S] = eig(correlation);
S(S < 0) = 0; % due to numerical precision errors
A = U * sqrt(S);

%% Monte Carlo Simulation
P_total = zeros(N, NumSamples); % Power at each port
xi = sqrt(gamrnd(m, 1/m, [1, NumSamples])); % Common shadowing factor

for i = 1:mu
    % 1. i.i.d. standard complex Gaussian scattered field
    w = (randn(N, NumSamples) + 1j * randn(N, NumSamples)) / sqrt(2);
    
    % 2. Apply spatial correlation
    g_scattered_complex = A * w * sqrt(scatter_power_per_cluster);
    
    % 3. Add shadowed dominant component
    dominant_component_complex = xi .* rho_i;
    
    % 4. Combine and accumulate non-coherently
    h_cluster_i = g_scattered_complex + dominant_component_complex;
    P_total = P_total + abs(h_cluster_i).^2;
end

R_simulated = sqrt(P_total); % Envelopes

%% Save dataset
metadata.author   = 'Pedro M. R. Pereira';
metadata.date     = datetime("now", Format="uuuu-MM-dd");
metadata.license  = 'MIT License';
metadata.references = { ...
    % INCLUDE OUR REFERENCE AFTER PUBLICATION
    'F. J. Lopez-Martinez, J. F. Paris, and J. M. Romero-Jerez, "The kappa-mu Shadowed Fading Model: Unifying the kappa-mu and eta-mu Distributions," IEEE Trans. Veh. Technol., vol. 65, no. 12, pp. 9630–9641, Dec. 2016.', ...
    'W. Waqar, M. Shafiq, S. Salehi, M. Egan, and J. Thompson, "Deep Learning Enabled Slow Fluid Antenna Multiple Access," IEEE Commun. Lett., vol. 27, no. 3, pp. 861–865, 2023.' ...
};

metadata.description = 'Dataset generated for correlated kappa-mu shadowed fading model in a fluid antenna system.';

save_filename = sprintf('FAS_kappa%.2f_mu%d_m%d_W%.2f_N%d_NS%d.mat', ...
                        kappa, mu, m, W, N, NumSamples);

save(save_filename, 'R_simulated', 'kappa', 'mu', 'm', 'r_hat', ...
                    'N', 'W', 'NumSamples', 'A', 'rho_i', 'metadata');

fprintf('Dataset saved to: %s\n', save_filename);

%% Optional validation plot
port_to_plot = randi(N); % select one port;
env_port = R_simulated(port_to_plot, :);

r = 0:0.01:5;
gamma_vals = r.^2;
pdf_gamma = (mu^mu * m^m * (1+kappa)^mu) / (gamma(mu) * (mu*kappa + m)^m * r_hat) ...
          .* (gamma_vals./r_hat).^(mu-1) ...
          .* exp(-mu*(1+kappa)*gamma_vals./r_hat) ...
          .* hypergeom(m, mu, (mu^2 * kappa * (1+kappa) / (mu*kappa + m)) * (gamma_vals./r_hat));
env_theor = 2 .* r .* pdf_gamma;

figure('Color', 'w'); hold on;
plot(r, env_theor, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Theoretical PDF');
histogram(env_port, 'Normalization', 'pdf', 'NumBins', 120, ...
          'DisplayName', sprintf('Simulated (Port %d)', port_to_plot), ...
          'FaceColor', '#0072B2', 'EdgeColor', 'none', 'FaceAlpha', 0.8);
title(sprintf('\\kappa-\\mu Shadowed Fading (\\kappa=%.2f, \\mu=%d, m=%d)', kappa, mu, m));
xlabel('Envelope Magnitude'); ylabel('PDF'); xlim([0, 4]); grid on; legend show;
