## Coset Division

Coset division is a mathematical operation used in the context of polynomial arithmetic. this function is useful whenever bivariate polynimal is vanishes on two roots of unity set. 
for example if 
$$W_X={\lbrace w_x^i \rbrace }_0^n$$
and 
$$W_Y={\lbrace w_y^i \rbrace }_0^m$$

$$P(W_X,W_Y)=0$$

then we can say

$$P(x,y)=Q_x(x,y)(x^n-1)+Q_y(x,y)(y^m-1)$$

it simply equal to say that the polynomial is equal to zero on $$W_Y$$ **AND** $$W_X$$.
please consider that if we wanted to say that the polynomial is zero on $$W_Y$$ **OR** $$W_X$$. 
this equation would be such :


$$P(x,y)=Q_{x,y}(x,y)(x^n-1)(y^m-1)$$


### Implimentation 

Current implementation has these two situation :
1. **$$p\in\mathbb{F}_{(2n-1,2m-1)}[X,Y]$$**


2. **$$p\in\mathbb{F}_{(2n-1, \alpha m-1)}[X,Y] ; \alpha>2$$**


