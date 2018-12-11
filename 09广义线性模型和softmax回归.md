[TOC]

到目前为止，我们看过了回归的案例，也看了一个分类案例。在回归的案例中，我们得到的函数是 $y|x; \theta ∼ N (\mu, \sigma^2)$；而分类的案例中，函数是 $y|x; \theta ∼ Bernoulli(\phi)$，这里面的$\mu$ 和 φ 分别是 x 和 θ 的某种函数。在本节，我们会发现这两种方法都是一个更广泛使用的模型的特例，这种更广泛使用的模型就叫做广义线性模型。我们还会讲一下广义线性模型中的其他模型是如何推出的，以及如何应用到其他的分类和回归问题上。

# 1、广义线性模型 （Generalized Linear Models）

## 1.1、指数族 （The exponential family）

在学习 GLMs 之前，我们要先定义一下指数组分布（exponential family distributions）。如果一个分布能用下面的方式来写出来，我们就说这类分布属于指数族：

$ p(y;\eta) =b(y)exp(\eta^TT(y)-a(\eta)) \text{(6)}$

上面的式子中，η 叫做此分布的自然参数（natural parameter，也叫典范参数 canonical parameter） ； $T(y)$ 叫做充分统计量（sufficient statistic），我们目前用的这些分布中通常 $T (y) = y$；而 $a(\eta)$ 是一个对数分割函数（log partition function）。$e^{−a(\eta)}$ 这个量本质上扮演了归一化常数（normalization constant）的角色，也就是确保 $p(y; \eta)$ 的总和或者积分等于1。

对 $T$, A 和 B 的固定选择，就定义了一个用 η 进行参数化的分布族（family，或者叫集 set）；通过改变$$\eta$$，我们就能得到这个分布族中的不同分布。
现在咱们看到的伯努利（Bernoulli）分布和高斯（Gaussian）分布就都属于指数分布族。

## 1.2、伯努利分布

伯努利分布的均值是φ，也写作 $Bernoulli(\phi)$，确定的分布是 $y ∈ {0, 1}$，因此有 $p(y = 1; \phi) = \phi$; $p(y = 0;\phi) = 1−\phi$。这时候只要修改φ，就能得到一系列不同均值的伯努利分布了。现在我们展示的通过修改φ,而得到的这种伯努利分布，就属于指数分布族；也就是说，只要给定一组 $T$，A 和 B，就可以用上面的等式(6)来确定一组特定的伯努利分布了。
我们这样来写伯努利分布：

$\begin{aligned}
p(y;\phi) & = \phi ^y(1-\phi)^{1-y}\\& = exp(\log (\phi ^y(1-\phi)^{1-y}))\\& = exp(y \log \phi + (1-y)\log(1-\phi))\\
& = exp( (log (\frac {\phi}{1-\phi}))y+\log (1-\phi) )\\
\end{aligned}$

因此，自然参数（natural parameter）就给出了，即 $\eta = log (\frac   \phi {1 − \phi})$ 。 很有趣的是，如果我们翻转这个定义，用η 来解 φ 就会得到 $\phi = 1/ (1 + e^{−\eta} )$。这正好就是之前我们刚刚见到过的 S型函数（sigmoid function）！ 在我们把逻辑回归作为一种广义线性模型（GLM）的时候还会遇到这个情况。

$\begin{aligned}
T(y) &= y \\
a( \eta) & = - \log (1- \phi) \\
& = \log {(1+ e^ \eta)}\\
b(y)&=1
\end{aligned}$

上面这组式子就表明了伯努利分布可以写成等式(6)的形式，使用一组合适的$T$， A 和 B。

## 1.3、高斯分布

接下来就看看高斯分布吧。还记得吧，在推导线性回归的时候，$\sigma%^2$ 的值对我们最终选择的 θ 和 $h_\theta(x)$ 都没有影响。所以我们可以给 $\sigma^2$ 取一个任意值。为了简化推导过程，就令$\sigma^2 = 1$。然后就有了下面的等式：

$\begin{aligned}
p(y;\mu) &= \frac 1{\sqrt{2\pi}} exp (- \frac  12 (y-\mu)^2) \\
& =  \frac 1{\sqrt{2\pi}} exp (- \frac  12 y^2) \times exp (\mu y -\frac  12 \mu^2) \\
\end{aligned}$

注：如果我们把 $\sigma^2$ 留作一个变量，高斯分布就也可以表达成指数分布的形式，其中 $\eta ∈ R^2$ 就是一个二维向量，同时依赖 $\mu$ 和 $\sigma$。然而，对于广义线性模型GLMs方面的用途， $\sigma^2$ 参数就也可以看成是对指数分布族的更泛化的定义： $p(y; \eta, \tau ) = b(a, \tau ) exp((\eta^T T (y) − a(\eta))/c(\tau))$。这里面的$\tau$ 叫做**分散度参数（dispersion parameter）**，对于高斯分布， $c(\tau) = \sigma^2$ ；不过上文中我们已经进行了简化，所以针对我们要考虑的各种案例，就不需要再进行更加泛化的定义了。
这样，我们就可以看出来高斯分布是属于指数分布族的，可以写成下面这样：

$\begin{aligned}
\eta & = \mu \\
T(y) & = y \\
a(\eta) & = \mu ^2 /2\\
& = \eta ^2 /2\\
b(y) & = (1/ \sqrt {2\pi })exp(-y^2/2)
\end{aligned}$

指数分布族里面还有很多其他的分布：例如多项式分布（multinomial），这个稍后我们会看到；泊松分布（Poisson），用于对计数类数据进行建模，后面再问题集里面也会看到；$\gamma$和指数分布（the gamma and the exponential），这个用于对连续的、非负的随机变量进行建模，例如时间间隔；$\beta$和狄利克雷分布（the beta and the Dirichlet），这个是用于概率的分布；还有很多啦。在下一节里面，我们就来讲一讲对于建模的一个更通用的“方案”，其中的y (给定 x 和 θ) 可以是上面这些分布中的任意一种。

# 2、构建广义线性模型（Constructing GLMs）

设想你要构建一个模型，来估计在给定的某个小时内来到你商店的顾客人数（或者是你的网站的页面访问次数），基于某些确定的特征 x ，例如商店的促销、最近的广告、天气、今天周几啊等等。我们已经知道泊松分布（Poisson distribution）通常能适合用来对访客数目进行建模。知道了这个之后，怎么来建立一个模型来解决咱们这个具体问题呢？非常幸运的是，泊松分布是属于指数分布族的一个分部，所以我们可以使用一个广义线性模型（Generalized Linear Model，缩写为 GLM）。在本节，我们讲一种对刚刚这类问题来构建广义线性模型的方法。

进一步泛化，设想一个分类或者回归问题，要预测一些随机变量 y 的值，作为 x 的一个函数。要导出适用于这个问题的广义线性模型，就要对我们的模型、给定 x 下 y 的条件分布来做出以下三个假设：

1. $y | x; \theta ∼ Exponential Family(\eta)$，即给定 x 和 θ, y 的分布属于指数分布族，是一个参数为 η 的指数分布。
2. 给定 x，目的是要预测对应这个给定 x 的 $T(y)$ 的期望值。咱们的例子中绝大部分情况都是 $T(y) = y$，这也就意味着我们的学习假设 h 输出的预测值 $h(x)$ 要满足 $h(x) = E[y|x]$。 (注意，这个假设通过对 $h_\theta(x)$ 的选择而满足，在逻辑回归和线性回归中都是如此。例如在逻辑回归中， $h_\theta (x) = [p (y = 1|x; \theta)] =[ 0 \times p (y = 0|x; \theta)+1\times p(y = 1|x;\theta)] = E[y|x;\theta]$。译者注：这里的$E[y|x$]应该就是对给定x时的y值的期望的意思。)
3. 自然参数$$\eta$$和输入值 x 是线性相关的，$\eta = \theta^T x$，或者如果 $$\eta$$ 是有值的向量，则有$\eta_i = \theta_i^T x$。

上面的几个假设中，第三个可能看上去证明得最差，所以也更适合把这第三个假设看作是一个我们在设计广义线性模型时候的一种 **“设计选择 design choice”**，而不是一个假设。那么这三个假设/设计，就可以用来推导出一个非常合适的学习算法类别，也就是广义线性模型 GLMs，这个模型有很多特别友好又理想的性质，比如很容易学习。
此外，这类模型对一些关于 y 的分布的不同类型建模来说通常效率都很高；例如，我们下面就将要简单介绍一些逻辑回归以及普通最小二乘法这两者如何作为广义线性模型来推出。

## 2.1、普通最小二乘法（Ordinary Least Squares）

我们这一节要讲的是普通最小二乘法实际上是广义线性模型中的一种特例，设想如下的背景设置：目标变量 y（在广义线性模型的术语也叫做响应变量response variable）是连续的，然后我们将给定 x 的 y 的分布以高斯分布 $N(\mu, \tau^2)$ 来建模，其中 $\mu$ 可以是依赖 x 的一个函数。这样，我们就让上面的指数分布族的$(\eta)$分布成为了一个高斯分布。在前面内容中我们提到过，在把高斯分布写成指数分布族的分布的时候，有$\mu = \eta​$。所以就能得到下面的等式：

$
\begin{aligned}
h_\theta(x)& = E[y|x;\theta] \\
& = \mu \\
& = \eta \\
& = \theta^Tx\\
\end{aligned}$

第一行的等式是基于假设(2)；第二个等式是基于定理当 $y|x; \theta ∼ N (\mu, \sigma ^2)$，则 y 的期望就是 $\mu$ ；第三个等式是基于假设(1)，以及之前我们此前将高斯分布写成指数族分布的时候推导出来的性质 $\mu = \eta$；最后一个等式就是基于假设(3)。

## 2.2、逻辑回归（Logistic Regression）

接下来咱们再来看看逻辑回归。这里咱们还是看看二值化分类问题，也就是 $y ∈ {0, 1}$。给定了y 是一个二选一的值，那么很自然就选择伯努利分布（Bernoulli distribution）来对给定 x 的 y 的分布进行建模了。在我们把伯努利分布写成一种指数族分布的时候，有 $\phi = 1/ (1 + e^{−\eta})$。另外还要注意的是，如果有 $y|x; \theta ∼ Bernoulli(\phi)$，那么 $E [y|x; \theta] = \phi​$。所以就跟刚刚推导普通最小二乘法的过程类似，有以下等式：

$
\begin{aligned}
h_\theta(x)& = E[y|x;\theta] \\
& = \phi \\
& = 1/(1+ e^{-\eta}) \\
& = 1/(1+ e^{-\theta^Tx})\\
\end{aligned}$

所以，上面的等式就给了给了假设函数的形式：$h_\theta(x) = 1/ (1 + e−\theta^T x)$。如果你之前好奇咱们是怎么想出来逻辑回归的函数为$1/ (1 + e^{−z} )$，这个就是一种解答：一旦我们假设以 x 为条件的 y 的分布是伯努利分布，那么根据广义线性模型和指数分布族的定义，就会得出这个式子。

再解释一点术语，这里给出分布均值的函数 $g$ 是一个关于自然参数的函数，$g(\eta) = E[T(y); \eta]$，这个函数也叫做规范响应函数（canonical response function），它的反函数 $g^{−1}$ 叫做规范链接函数（canonical link function）。因此，对于高斯分布来说，它的规范响应函数正好就是识别函数（identify function）；而对于伯努利分布来说，它的规范响应函数则是逻辑函数（logistic function）。

> 注：很多教科书用 $g$ 表示链接函数，而用反函数$g^{−1}$ 来表示响应函数；但是咱们这里用的是反过来的，这是继承了早期的机器学习中的用法，我们这样使用和后续的其他课程能够更好地衔接起来。

## 2.3、Softmax 回归

咱们再来看一个广义线性模型的例子吧。设想有这样的一个分类问题，其中响应变量 y 的取值可以是 k 个值当中的任意一个，也就是 $y ∈ \{1, 2, ..., k\}$。例如，我们这次要进行的分类就比把邮件分成垃圾邮件和正常邮件两类这种二值化分类要更加复杂一些，比如可能是要分成三类，例如垃圾邮件、个人邮件、工作相关邮件。这样响应变量依然还是离散的，但取值就不只有两个了。因此咱们就用多项式分布（multinomial distribution）来进行建模。
下面咱们就通过这种多项式分布来推出一个广义线性模型。要实现这一目的，首先还是要把多项式分布也用指数族分布来进行描述。
要对一个可能有 k 个不同输出值的多项式进行参数化，就可以用 k 个参数 $\phi_1,...,\phi_ k$ 来对应各自输出值的概率。不过这么多参数可能太多了，形式上也太麻烦，他们也未必都是互相独立的（比如对于任意一个$\phi_ i$中的值来说，只要知道其他的 $k-1$ 个值，就能知道这最后一个了，因为总和等于1，也就是$\sum^k_{i=1} \phi_i = 1$）。
所以咱们就去掉一个参数，只用 $k-1$ 个：$\phi_1,...,\phi_ {k-1}$  来对多项式进行参数化，其中$\phi_i = p (y = i; \phi)，p (y = k; \phi) = 1 −\sum ^{k−1}_{i=1}\phi_ i$。为了表述起来方便，我们还要设 $\phi_k = 1 − \sum_{i=1}^{k−1} \phi_i$，但一定要注意，这个并不是一个参数，而是完全由其他的 $k-1$ 个参数来确定的。要把一个多项式表达成为指数组分布，还要按照下面的方式定义一个 $T (y) ∈ R^{k−1}$:

$
T(1)=\left[
​    \begin{array}{cc|c}
​      1\\
​      0\\
​	  0\\
​	  \vdots \\
​	  0\\
​    \end{array}
\right],
T(2)=\left[
​    \begin{array}{cc|c}
​      0\\
​      1\\
​	  0\\
​	  \vdots \\
​	  0\\
​    \end{array}
\right],
T(3)=\left[
​    \begin{array}{cc|c}
​      0\\
​      0\\
​	  1\\
​	  \vdots \\
​	  0\\
​    \end{array}
\right],
T(k-1)=\left[
​    \begin{array}{cc|c}
​      0\\
​      0\\
​	  0\\
​	  \vdots \\
​	  1\\
​    \end{array}
\right],
T(k)=\left[
​    \begin{array}{cc|c}
​      0\\
​      0\\
​	  0\\
​	  \vdots \\
​	  0\\
​    \end{array}
\right]
$

这次和之前的样例都不一样了，就是不再有 $T(y) = y$；然后，$T(y)$ 现在是一个 $k – 1$ 维的向量，而不是一个实数了。向量 $T(y)$ 中的第 i 个元素写成$(T(y))_i$ 。
现在介绍一种非常有用的记号。指示函数（indicator function）$1\{\cdot  \}$，如果参数为真，则等于1；反之则等于0（$1{True} = 1, 1{False} = 0$）。例如1{2 = 3} = 0, 而1{3 = 5 − 2} = 1。所以我们可以把T (y) 和 y 的关系写成  $(T(y))i = 1{y = i}$。（往下继续阅读之前，一定要确保你理解了这里的表达式为真！）在此基础上，就有了$E[(T(y))i] = P (y = i) = \phi_i$。
现在一切就绪，可以把多项式写成指数族分布了。
写出来如下所示：



![](https://raw.githubusercontent.com/Kivy-CN/Stanford-CS-229-CN/master/img/cs229note1f8.png)

其中：

$\begin{aligned}
\eta &= \left[
​    \begin{array}{cc|c}
​      \log (\phi _1/\phi _k)\\
​      \log (\phi _2/\phi _k)\\
​	  \vdots \\
​	  \log (\phi _{k-1}/\phi _k)\\
​    \end{array}
\right]\\
a(\eta) &= -\log (\phi _k)\\
b(y) &= 1\\
\end{aligned}$

这样咱们就把多项式方程作为一个指数族分布来写了出来。
与 $i (for  : i = 1, ..., k)$对应的链接函数为：

$ \eta_i =\log \frac  {\phi_i}{\phi_k}$

为了方便起见，我们再定义 $\eta_k = \log (\phi_k/\phi_k) = 0$。对链接函数取反函数然后推导出响应函数，就得到了下面的等式：

$\begin{aligned}
e^{\eta_i} &= \frac {\phi_i}{\phi_k}\\
\phi_k e^{\eta_i} &= \phi_i \text{(7)}\\
\phi_k  \sum^k_{i=1} e^{\eta_i}&= \sum^k_{i=1}\phi_i= 1\\
\end{aligned}$

这就说明了$\phi_k = \frac  1 {\sum^k_{i=1} e^{\eta_i}}$，然后可以把这个关系代入回到等式(7)，这样就得到了响应函数：

$ \phi_i = \frac  { e^{\eta_i} }{ \sum^k_{j=1} e^{\eta_j}}$

上面这个函数从η 映射到了φ，称为 Softmax 函数。要完成我们的建模，还要用到前文提到的假设(3)，也就是 $\eta_i$ 是一个 x 的线性函数。所以就有了 $\eta_= \theta_i^Tx (for:i = 1, ..., k − 1)$，其中的 $\theta_1, ..., \theta_{k−1} ∈ R^{n+1}$ 就是我们建模的参数。为了表述方便，我们这里还是定义$\theta_k = 0$，这样就有 $\eta_k = \theta_k^T x = 0$，跟前文提到的相符。因此，我们的模型假设了给定 x 的 y 的条件分布为：

$\begin{aligned}
p(y=i|x;\theta) &=  \phi_i \\
&= \frac {e^{\eta_i}}{\sum^k_{j=1}e^{\eta_j}}\\
&=\frac {e^{\theta_i^Tx}}{\sum^k_{j=1}e^{\theta_j^Tx}}\text{(8)}\\
\end{aligned}$



这个适用于解决 $y ∈\{1, ..., k\}$ 的分类问题的模型，就叫做 Softmax 回归。 这种回归是对逻辑回归的一种扩展泛化。
假设（hypothesis） h 则如下所示:

$\begin{aligned}
h_\theta (x) &= E[T(y)|x;\theta]\\
&= E \left[
​    \begin{array}{cc|c}
​      1(y=1)\\
​      1(y=2)\\
​	  \vdots \\
​	  1(y=k-1)\\
​    \end{array}|x;\theta
\right]\\
&= E \left[
​    \begin{array}{cc|c}
​      \phi_1\\
​      \phi_2\\
​	  \vdots \\
​	  \phi_{k-1}\\
​    \end{array}
\right]\\
&= E \left[
​    \begin{array}{cc|c}
​      \frac {exp(\theta_1^Tx)}{\sum^k_{j=1}exp(\theta_j^Tx)} \\
​      \frac {exp(\theta_2^Tx)}{\sum^k_{j=1}exp(\theta_j^Tx)} \\
​	  \vdots \\
​	  \frac {exp(\theta_{k-1}^Tx)}{\sum^k_{j=1}exp(\theta_j^Tx)} \\
​    \end{array}
\right]\\
\end{aligned}$



也就是说，我们的假设函数会对每一个 $i = 1,...,k$ ，给出 $p (y = y|x; \theta)$ 概率的估计值。（虽然咱们在前面假设的这个 $h_\theta(x)$ 只有 $k-1$ 维，但很明显 $p (y = y|x; \theta)$ 可以通过用 1 减去其他所有项目概率的和来得到，即$1− \sum^{k-1}_{i=1}\phi_i$。）

最后，咱们再来讲一下参数拟合。和我们之前对普通最小二乘线性回归和逻辑回归的原始推导类似，如果咱们有一个有 m 个训练样本的训练集 $\{(x^{(i)}, y^{(i)}); i = 1, ..., m\}$，然后要研究这个模型的参数 $$\theta$$ ，我们可以先写出其似然函数的对数：

$\begin{aligned}
l(\theta)& =\sum^m_{i=1} \log p(y^{(i)}|x^{(i)};\theta)\\
&= \sum^m_{i=1} \prod ^k_{l=1}(\frac {e^{\theta_l^Tx^{(i)}}}{\sum^k_{j=1} e^{\theta_j^T x^{(i)}}})^{1(y^{(i)}=l)}\\
\end{aligned}$



要得到上面等式的第二行，要用到等式(8)中的设定 $p(y|x; \theta)$。现在就可以通过对 $l(\theta)$ 取最大值得到的 θ 而得到对参数的最大似然估计，使用的方法就可以用梯度上升法或者牛顿法了。

