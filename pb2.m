%% Problem 2 Estimating GARCH Models & Computing Volatility Forecasts
%% a
% compute Engle's LM test for ARCH effects for 5 & 10 lags
em = rm - mean(rm);
es = rs - mean(rs);
[hm,pm,fStatm,critm] = archtest(em,'Lags',[3,6]);
[hs,ps,fStats,crits] = archtest(es,'Lags',[3,6]);
%% b
% estimate ARCH & GARCH
% i
toestmdl_arch = garch(0,3);
estmdl_rm = estimate(toestmdl_arch,rm);
estmdl_rs = estimate(toestmdl_arch,rs);

% ii
res_rm=infer(estmdl_rm,rm);
res_rs=infer(estmdl_rs,rs);
abs_rm=abs(rm);
abs_rs=abs(rs);
% plot(sqrt(res_rm)); hold on; plot(abs_rm);
% legend('Conditional Volatility (ARCH) of Microsoft','Microsoft Absolute Return');
% figure;
% plot(sqrt(res_rs)); hold on; plot(abs_rs);
% legend('Conditional Volatility (ARCH) of S&P','S&P Absolute Return');
% figure;
plot(sqrt(res_rm)); 
legend('Conditional Volatility (ARCH) of Microsoft');
figure;
plot(abs_rm);
legend('Microsoft Absolute Return');
figure;
plot(sqrt(res_rs)); 
legend('Conditional Volatility (ARCH) of S&P');
figure;
plot(abs_rs);
legend('S&P Absolute Return');
figure;
% iii

toestmdl_garch = garch(1,1);
estmdl_rm_garch = estimate(toestmdl_garch,rm);
estmdl_rs_garch = estimate(toestmdl_garch,rs);

res_rm_garch=infer(estmdl_rm_garch,rm);
res_rs_garch=infer(estmdl_rs_garch,rs);
abs_rm_garch=abs(rm);
abs_rs_garch=abs(rs);
% plot(sqrt(res_rm_garch)); hold on; plot(abs_rm_garch);
% legend('Conditional Volatility (GARCH) of Microsoft','Microsoft Absolute Return');
% figure;
% plot(sqrt(res_rs_garch)); hold on; plot(abs_rs_garch);
% legend('Conditional Volatility (GARCH) of S&P','S&P Absolute Return');

plot(sqrt(res_rm_garch)); 
legend('Conditional Volatility (GARCH) of Microsoft');
figure;
plot(abs_rm_garch);
legend('Microsoft Absolute Return');
figure;

plot(sqrt(res_rs_garch)); 
legend('Conditional Volatility (GARCH) of S&P');
figure;
plot(abs_rs_garch);
legend('S&P Absolute Return');
figure;

%% c
f100ARCH_m = forecast(estmdl_rm,100,'Y0',rm);
f100GARCH_m = forecast(estmdl_rm_garch,100,'Y0',rm);
figure;
plot(sqrt(f100ARCH_m)); hold on; plot(sqrt(f100GARCH_m)); hold on;...
    plot(sqrt(estmdl_rm.UnconditionalVariance)*ones(100,1)); hold on;...
    plot(sqrt(estmdl_rm_garch.UnconditionalVariance)*ones(100,1));
legend('ARCH(3)','GARCH(1,1)','uncon vol ARCH','uncon vol GARCH');
title('Microsoft Conditional Volatility Forecasting');

f100ARCH_s = forecast(estmdl_rs,100,'Y0',rs);
f100GARCH_s = forecast(estmdl_rs_garch,100,'Y0',rs);
figure;
plot(sqrt(f100ARCH_s)); hold on; plot(sqrt(f100GARCH_s)); hold on;...
 plot(sqrt(estmdl_rs.UnconditionalVariance)*ones(100,1)); hold on;...
    plot(sqrt(estmdl_rs_garch.UnconditionalVariance)*ones(100,1));
legend('ARCH(3)','GARCH(1,1)','uncon vol ARCH','uncon vol GARCH');
title('S&P Conditional Volatility Forecasting');
%% Problem 3 EWMA Covariances ad Correlations
%% a
window=20; covar=zeros(1,length(rm)-20); corre=zeros(1,length(rm)-20);
for i=1:length(rm)-window
    temp=cov(rm(i:i+window-1),rs(i:i+window-1));
    covar(i)=temp(1,2); 
    corre(i)=corr(rm(i:i+window-1),rs(i:i+window-1));
end
figure;plot(covar);
title('Rolling Covariance');
figure;plot(corre);
title('Rolling Correlation');
%% b & c
% data=[rm,rs];
% lambda=0.94;
% [r,c]         = size(data);
% data_mwb      = data-repmat(mean(data,1),r,1);
% lambdavec     = lambda.^(0:1:r-1)';
% data_tilde    = repmat(sqrt(lambdavec),1,c).*data_mwb;
% 
% cov_ewma      = 1/sum(lambdavec)*(data_tilde'*data_tilde);
% corr_ewma     = zeros(c);
% vola_ewma     = zeros(c,1);
% 
% for i = 1:c
%     for j = 1:c
%         corr_ewma(i,j) = cov_ewma(i,j)/sqrt(cov_ewma(i,i)*cov_ewma(j,j));
%     end
%     vola_ewma(i) = sqrt(cov_ewma(i,i));
% end

T=4027;
a=rm-mean(rm);
b=rs-mean(rs);
y=[a b];
EWMA = zeros(T,3);	
lambda = 0.94;
S = cov(y);	
EWMA(1,:) = S([1,4,2]);	
for i = 2:T	
	S = lambda * S  + (1-lambda) * y(i,:)' * y(i,:);	
	EWMA(i,:) = S([1,4,2]);	
end
EWMArho = EWMA(:,3) ./ sqrt(EWMA(:,1) .* EWMA(:,2));
EWMA_covariance_matrix=EWMA(:,3);
plot(EWMA_covariance_matrix);
title('EWMA Conditional Covariance');
figure;
plot(EWMArho);
title('EWMA Conditional Correlation');
figure;
plot(corre);
title('Rolling Correlation');

%% d
%  100-h-step-ahead
ahead =100 ;
sigma12 = zeros(ahead,1) ;
cond_corr = zeros(ahead,1) ;
for i=1:ahead
 temp =lambda*ones(2,2).* temp ;
sigma12(i)=temp(1,2);
cond_corr(i)=sigma12(i)/(temp(1,1)*temp(2,2));
end

figure;
plot(sigma12)
title('Contional Covariance')

figure;
plot(cond_corr)
title('Contional Correlation')


