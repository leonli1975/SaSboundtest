%% demo_step1_step2.m —— 用实验用 MAT 走一遍 Step1（去斜/距离像/Range-Doppler）与 Step2（逐单元 SαS α）
% 只依赖基础 MATLAB（不需要 Signal Processing / Statistics 工具箱）
% 数据：T2exp\MTDSP2025_HH_413020540.mat
clear; clc;
f = 'MTDSP2025_HH_413020540.mat';
if ~isfile(f), f = fullfile('T2exp', f); end
S = load(f);
fprintf('文件 %s\n', f);
fprintf('  pc    : %s  %s\n', mat2str(size(S.pc)), class(S.pc));
fprintf('  方位  : %.2f ~ %.2f deg\n', min(S.az_deg), max(S.az_deg));
fprintf('  距离  : %.0f ~ %.0f m（%d 个单元，步长 %.1f m）\n', ...
        S.range_m(1), S.range_m(end), numel(S.range_m), S.range_m(2)-S.range_m(1));
fprintf('  极化 %s | 工作模式 %s | 海况 %.0f | 波高 %.2f m | 风速 %.1f m/s\n', ...
        S.meta.polarization, S.meta.mode, S.meta.sea_state, S.meta.wave_height_m, S.meta.wind_speed_ms);
fprintf('  杂波单元 %d | 目标单元 %d | 强散射单元 %d | 目标距离中位 %.0f m\n', ...
        numel(S.clutter_cells), numel(S.target_cells), numel(S.strong_cells), S.meta.target_range_med_m);

%% ================= Step 1：去斜 + 距离像 + Range-Doppler =================
fs = 60e6; f0 = 15e6; k = 3.12e12; c = 299792458;
x  = double(S.if_raw);              % (Np,1760) 中频实信号（脉压前）
[Np, NR] = size(x);
t  = (0:NR-1)/fs;

% 1) 去直流 + 解析信号（FFT 法，替代 hilbert）
x = x - mean(x,2);
X = fft(x,[],2);
h = zeros(1,NR); h(1) = 1;
if mod(NR,2)==0, h(NR/2+1)=1; h(2:NR/2)=2; else, h(2:(NR+1)/2)=2; end
xa = ifft(X.*h,[],2);

% 2) 去斜：乘 exp(-j(2πf0 t + π k t²))  ->  单频，频率 f_b = k·τ
y  = xa .* exp(-1j*(2*pi*f0*t + pi*k*t.^2));

% 3) 距离剖面（相对波门起点）
P  = abs(fftshift(fft(y,[],2),2));
fb = fftshift((-NR/2:NR/2-1)*(fs/NR));
r_rel = c*fb/(2*k);

% 4) 5° 方位带 -> Range-Doppler
az  = S.az_deg(:);
a0  = quant(az, 0.4);
sel = az>=a0 & az<a0+5;
if sum(sel) < 64, sel = true(size(az)); end
Y  = P(sel,:) - mean(P(sel,:),1);
Y  = Y .* hann(size(Y,1));
RD = fftshift(fft(Y,[],1),1);
nD = size(RD,1); fd = fftshift((-floor(nD/2):ceil(nD/2)-1)'*(2000/nD));
fprintf('Step1: 方位带 %.1f~%.1f°，%d 脉冲，多普勒 ±%.0f Hz\n', a0, a0+5, sum(sel), max(abs(fd)));

figure('Name','Step1');
subplot(1,2,1);
plot(r_rel, 20*log10(mean(P,1)/max(mean(P,1))+eps)); grid on; xlim([-1200 1200]);
xlabel('相对距离 / m'); ylabel('dB'); title('去斜距离剖面（平均）');
subplot(1,2,2);
imagesc(r_rel, fd, 20*log10(abs(RD)/max(abs(RD(:)))+eps)); axis xy;
caxis([-45 0]); colorbar;
xlabel('相对距离 / m'); ylabel('多普勒 / Hz'); title('Range-Doppler（5° 方位带）');

%% ================= Step 2：逐单元 SαS α（log-ECF 回归） =================
cc  = S.clutter_cells(:); cc = cc(cc>=1 & cc<=size(S.pc,2));
a1  = quant(az, 0.6); sel2 = az>=a1 & az<a1+5;
fprintf('Step2: 方位带 %.1f~%.1f°，%d 脉冲\n', a1, a1+5, sum(sel2));
idx = cc(round(linspace(1, numel(cc), min(24, numel(cc)))));
al = nan(numel(idx),1);
for ii = 1:numel(idx)
    v = real(S.pc(sel2, idx(ii)));
    al(ii) = alpha_log_ecf(v);
end
fprintf('  α 中位 %.2f（%.2f~%.2f）\n', median(al), min(al), max(al));
figure('Name','Step2');
plot(S.range_m(idx), al, 'o-'); grid on; ylim([0.9 2.05]);
xlabel('距离 / m'); ylabel('\alpha'); title(sprintf('\\alpha 中位 %.2f', median(al)));

%% ================= 局部函数 =================
function q = quant(x, p)
x = sort(x(~isnan(x))); n = numel(x);
q = x(max(1, min(n, round(p*(n-1))+1)));
end

function a = alpha_log_ecf(v)
% 对称 α-stable：φ(t)=E[cos(tX)]=exp(-γ|t|^α) => ln(-ln φ) = lnγ + α ln t
v = v(isfinite(v)); v = v - median(v);
s = std(v); if s<=0 || numel(v)<200, a = NaN; return; end
v = v/s;
ts = linspace(0.05, 2.7, 24);
phi = arrayfun(@(t) mean(cos(t*v)), ts);
ok = phi > 0.15 & phi < 0.995;
if sum(ok) < 5, a = NaN; return; end
y  = log(-log(phi(ok)));
lt = log(ts(ok));
p  = polyfit(lt, y, 1);
a  = p(1);
end
