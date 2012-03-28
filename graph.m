function graph(A)
%GRAPH Draws a graph of the adjacency matrix A.
A=A>0;
[m,n]=size(A);
if m~=n
	error('Adjacency matrix must be square.')
end
vertices=exp(j*2*pi/n.*(0:n-1));
vertices=vertices(:);
x=real(vertices);
y=imag(vertices);
gplot(A,[x,y])
axis([-1.3 1.3 -1.3 1.3])
for k=0:n-1
	h=text(1.1*cos(2*k*pi/n),1.1*sin(2*k*pi/n),['P',num2str(k+1)]);
	set(h,'horiz','center')
end
grid off
axis off