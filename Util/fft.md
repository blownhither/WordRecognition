### DFT:
$$X(k) = \sum_{n=0}^{N-1}x(n)W_N^{nk} \ \ (k = 0,1...N-1)$$
$$W_N=e^{-j\frac{2\pi}{N}}$$
then X(k) is relative intensity on frequency [-fs/2, fs/2-fs/N], interval fs/N  
This algorithm is $O(n^2)$.  

### FFT:
let $N = 2^L$, we split x as follow:
$$x(2r)=x_1(r)$$
$$x(2r+1)=x_2(r)$$
$$r=0, ..., N/2-1$$
therefore, 
$$X(k)=DFT[x(n)]=\sum x(n)W_N^{nk}$$
$$=\sum x(2r)W_n^{2rk} + \sum x(2r+1)W_N^{(2r+1)k}$$
$$=\sum_{r=0}^{N/2-1} x_1(r)W_{N/2}^{rk} + W^k_N \sum_{r=0}^{N/2-1}x_2(r)W_{N/2}^{rk}$$
further,
$$X(k)=X_1(k) + W^k_N X_2(k),\ \ \ k=0...N/2-1$$
$$X(k)=X_i(k')-W_N^{k'},\ \ \ k'=0...N/2-1,\ \ \ (k'=k-N/2)$$
finally,
$$X_N(k)=X'_{N/2}(k')+W_N^kX_{N/2}''(k''),\ \ k = 0...N/2-1$$
$$X_N(k)=X'_{N/2}(k')-W_N^kX_{N/2}''(k''),\ \ k = N/2...N-1$$
where $X'(k')$ is even branch and $X''(k'')$ is odd branch.  
This algorithm is $O(2N\log N)$