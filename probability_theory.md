# Read these resourses before this markdown. This thing is just a reminder of what you will learn in the following sourses:
* "Deep Learning" by Ian Goodfellow et al. Chapter "Probability theory"
* Or [this](https://towardsdatascience.com/probability-fundamentals-of-machine-learning-part-1-a156b4703e69#:~:text=Fundamentals%20of%20Machine%20Learning%20(Part%201)&text=Probability%20theory%20is%20a%20mathematical,of%20many%20machine%20learning%20algorithms).
* [This is just a quick introduction](https://medium.com/machine-learning-mindset/probability-theory-and-its-huge-importance-in-machine-learning-3a61b1601ccb)


## Probability.
Why do we need probabilities when we already have many other mathematical tools?  We have calculus to work with functions at infinite intervals. We have algebra for solving systems of equations, etc. The main problem is that we live in a chaotic universe where we can't measure all things accurately. Random events, unaccounted data affect our experiments, distorting them. That is, everywhere there is a certain uncertainty about the results and errors. This is where probability theory and statistics come to the fore.

There is a lot of definition of probability, consider the particular. We will toss a coin and get two results: heads or tails. Toss 1000 times, we get: 600 eagles and 400 tails. Then the probability of falling out of the eagle is $\frac{600}{1000}=0.6$, and the probability of dipping tails is $\frac{400}{1000}= 0.4$. Formally speaking, the probability of event $A$ is equal to the number of occurrences of this event $N_A$ in relation to the number of all events $N$ - $P(A) = \frac{N_A}{N}$.

For example, we take turns tossing two dice. Let the event $A={'in  sum  on bones  fell  10'}$ , find P(A)=?. Consider all possibilities when the sum will be equal to 10: (4,6),  (5,5),  (4,6). (4.6) i.e. $N_A=2$,  $N=6*6=36$. Then $P(A)=\frac{2}{36}=\frac{1}{18}$.

## Conditional probabilities.
Often we need to know not only the probability of an event, but the probability of an event when something has already happened. For example, the event 'tomorrow afternoon will be raining'. If you're asking this question tonight, there may be one chance. However, if you think about it tomorrow morning, the probability may be different. For example, if it is cloudy in the morning, then the probability of rain in the evening will increase (for example). This is a conditional probability. Denoted $P(A| B)$ is the probability of event $A$, provided that event $B$ has come true. Let's look at more examples: 
* What is the probability of rain if thunder thunders?
* What is the probability of rain if it is sunny now?
  
![img](https://imgur.com/A5xMRbA)

A slightly simplified image in the form of Euler diagrams, but it conveys the meaning. From the diagram you can understand that $P(rain \| thunder) = 1$, i.e. when there is thunder, there will always be rain. What about $P(rain|sunny)=\frac{P(rain  \space and  \space sunny)}{P(sunny)}$, i.e.  $P(A| B)=\frac{P(A,B)}{P(B)}$.

Events A and  B are called independent if $P(A,B)=P(A)P(B)$,or the same as $P(A)=P(A| B)$. In this diagram, all events are dependent, because if one thing happens, you can draw conclusions about whether another will happen. An example of independent events: weather and the number of apps on your phone ( number of apps doesn't depend on the weather).
### Basic properties of probability.
1. The probability of an impossible event is zero: $P(\emptyset)=0$
2. If event $A$ is included in $B$ , i.e. if event $A$ occurs, then event $B$ definitely occurred, then: $P(A)\le P(B)$
3. The probability of each $A$ event is from $0$ to $1$ .
4. $P(B\backslash A)=P(B)-P(A)$ is the event when $B$ has occurred and $A$ is not.
5. The probability of the opposite event to $A$ is: $P(\overline{A})=1-P(A)$
6. $P(A+B)=P(A)+P(B)-P(AB)=0$, then the events are called incompatible .
### Bayes' formula. 
We know that $P(A| B)=\frac{P(AB)}{P(B)}$. Let's find the probability of another $P(B| A)=?$ Using the conditional probability formula, we get: $\boxed{P(B| A)=\frac{P(BA)}{P(A)}}$ 

But since $P(AB)=P(BA)$, for they are the same thing, and $P(AB)=P(A| B)P(B)$, we obtain Bayes' formula:

$\boxed{P(B| A)=\frac{P(BA)}{P(A)}=\frac{P(AB)}{P(A)}=\frac{P(A| B)P(B)}{P(A)}}$

Bayes' formula helps to "rearrange cause and effect": to find the probability that the $B$ event was caused by the cause of $A$.

### Full probability formula.
Let $B_1,\dots,B_n$ be incompatible events, i.e. those that cannot occur at the same time. For example, you only take 1 tickets out of 15 on an exam. You can't get two tickets at the same time. An additional condition for these events is that together they form all possible outcomes: $P(B_1)+\dots+P(B_n)=1$. This means that at least one event will happen. For example, on the same exam, you will in any case get one of the tickets. The event $B_1,\dots,B_nB$  are said to form a __full group of mismatched events__.

Let's still have an $A$ event - for example, you came across a topic about conditional probabilities in the ticket. Consider the probabilities $P(A,B_{1}),\dots,P(A,B_{n})$ — i.e. the probability that you got the $i$ ticket and the topic about conditional probabilities. Since you will definitely get some kind of ticket, then: $\boxed{P(A)=P(A,B_{1})+\dots+P(A,B_{n})}$ Let's write each probability according to the rule of conditional probability: 

$P(A)=P(A,B_{1})+\dots+P(A,B_{n})=P(B_{1})P(A| B_{1})+\dots+P(B_{n})P(A| B_{n})=\sum_{i=1}^n P(B_{i})P(A| B_{i})$

__is full probability formula__.

It can be used to improve Bayes' formula:$\boxed{P(B_{i}| A)=\frac{P(B_{i})P(A| B_{i})}{P(A)}=\frac{P(B_{i})P(A| B_{i})}{\sum_{i=1}^n P(B_{i})P(A| B_{i})}}$

Probabilities $P(B_{i}| A)$ is called __a posteriori probabilities__ (i.e., probabilities that are slightly refined because an $A$ event occurred, and events $P(B_{i})$ are a priori probabilities (probabilities that are known prior to the $A$ experiment).

### A naïve Bayesian classifier.
Let's use the previous formulas for the classification task. Suppose we have a task of classifying texts into $C=\{c_{1},\dots,c_{n}\}$ classes.
What would you take as features? For example, what words appear in the texts. That is, let all the texts of the entire $N$ have unique words. We will give each word a number from $1$  to  $N$. The features of each text will be the number of words that are in the text. For example, $2$ words 'house', $3$ words 'cat', $0$ words 'anime'. Thus, we can represent each text as a length vector $N$: $(w_{1},...,w_{N})$, where $x_{i}$ is the number of words number $i$ in our text.
We have many such texts, so we can calculate different probabilities. 

For example:
* How many texts are there in each class $P(c)$
* What is the probability of encountering a word numbered $1$: $P(w_{1})$
*  How many times does a word under number $2$ in different classes of $P(w_{2}|c)$ occur? To do this, we take all texts containing the word $w_2$. Look at how many of these letters are marked with class $c_1$, how many with class $c_2$, etc.

In the classification problem, we have a text consisting of such-and-such words, i.e. it has features. Our task is to understand the $c$ text class. I.e. find the probabilities $P(c|w_1,\dots,w_N)$,
sort through all the $c$ and choose from them the class that gives the maximum probability. Formally: $\hat{c}=\underset{c}{argmax}P(c|w_1,\dots w_N)$, where  $\hat {c}$ is our prediction.

Let's convert this probability using Bayes' formula: 
$\boxed{P(c|w_1,\dots,w_N)=\frac{P(c,w_1,\dots w_N)}{P(w_1,\dots,w_N)}}$

Since we are maximizing by $c$ , and the probabilities of $w_1,\dots w_N$ remain the same for writing, then they can be removed from argmax.
$\boxed {P(c,w_1,\dots,w_N)=P(c)P(w_1,\dots,w_N|c)}$

The probability on the right is quite complicated. Let's make a "naïve" assumption: all words in texts appear independently of each other. That is, if there are 22 words 'cat' in the text, then the number of 'dogs' can be any. This is the word 'naïve' in the name of this algorithm.

$\boxed{P(c,w_1,\dots,w_N)=P(c)P(w_1|c)\dots P(w_N|c)}$

We can calculate all the probabilities on the right from the training sample. Then we get the final answer of the algorithm:

$\boxed{\hat{c}=\underset{c}{argmax}P(c)P(w_{1}|c)...P(w_{N}|c)}$

<hr>

You have to read about:
* What is probability distirbutions? What is continiuous distirbutions and what are discrete ones? What are the most common distirbutions? (Bernoulli distribution, Gauss distribution, Poisson distribution, and others). [Here](https://towardsdatascience.com/practical-guide-to-common-probability-distributions-in-machine-learning-487f6137625) you can find some of them. I would advice you to use wikipedia, it ma look diffucult but you would rather understand it better than left it because of diffuculties
* Maximum likelihood estimation. Basically wikipedia and "Deep learning" which I wrote about in the begining
* Entropy. Entropy is used not only in probability theory, but also in physics. But we'll look at information entropy. This is a measure of the uncertainty of some system. For example, let there be a stream of incoherent Russian letters (your answer on the exam). All symbols appear equally likely and the entropy of this system is maximum. But if we know that there is a stream not of incoherent letters, but of words of the Russian language. Entropy decreases, because we can make some estimates on which letter will appear next. And if there is a stream of not words, but well-composed texts. Entropy will decrease even more.

  Let's give a formal definition of entropy. Suppose we have a random experiment $X$ with $n$ possible answers, i.e. $X$ is a discrete random variable. The probability of each answer is $p_i$. Then the entropy is: $H(X)=\sum_{i=1}^{n} p_{i}log_{2}p_{i}$ (the more random and chaos is in the system, the more its entropy)
* Cross-entropy. Cross-entropy shows the quantitative difference between two probability distributions. It is defined as follows: $H(p;q)=-\underset{x}{\sum}p(x)log \space q(x)$
  
  This function can be used as a loss function (it is called the logistic loss function). For example, $q$ is a true distribution, it looks like a bunch of zeros on the wrong classes, and one unit on the right one. And $p$ is the result of our algorithm, where we say 'here to the 1th class the object refers with a probability of 0.2, but to the 2nd with a probability of 0.7, etc. We can differentiate this function and therefore we will be able to use different teaching methods.
* Finally about __Logistic Regression__. It is a very important ML classification algorithm and you must understand it deeply:
  * [wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
  * [Towards data science](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
  * [GeeksforGeeks](https://www.geeksforgeeks.org/understanding-logistic-regression/) 