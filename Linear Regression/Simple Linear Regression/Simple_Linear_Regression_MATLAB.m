clear all
close all
clc
m=5; c =10 ; % slope and intercepts
x= (-5:10)'; % x is a column vector
y=m*x+c; % y is a column vector
n=length(x); % number of data points
% let us generate n disturbing values from normal distribution with
% sigma=5 . That is, standard deviation =5
noise= 5*randn(n,1); % nx1 column vector
yd=y+noise;
A=[x ones(n,1)];
Alpha= inv(A'*A)*A'*yd;
ycap=A*Alpha;
plot(x,yd,'*') ; % plot scattered data points
hold on
plot(x, ycap); % plot the regression line
xlabel('independent variable x')
ylabel('dependent variable y')
ev =yd-ycap; % error vector.
% Verifying that error vector is orthogonal to column vectors of A
Check = A'*ev ;
% print check on screen
Format bank
Check
