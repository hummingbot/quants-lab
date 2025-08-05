# The Market Meanness Index – The Financial Hacker

[https://financial-hacker.com/the-market-meanness-index/](https://financial-hacker.com/the-market-meanness-index/)

![](https://financial-hacker.com/wp-content/uploads/2015/09/bear2.png)

This indicator can improve – sometimes even double – the profit expectancy of trend following systems. The **Market Meanness Index** tells whether the market is currently moving in or out of a “trending” regime. It can this way prevent losses by **false signals** of trend indicators. It is a purely statistical algorithm and not based on volatility, trends, or cycles of the price curve.

There are already several methods for differentiating trending and nontrending market regimes. Some of them are rumored to really work, at least occasionally. John Ehlers proposed the **Hilbert Transform** or a **Cycle / Trend decomposition**, Benoit Mandelbrot the **Hurst Exponent**. In comparison, the source code of the Market Meanness Index is relatively simple:

```
// Market Meanness Indexdouble MMI(double *Data,int Length){
  double m = Median(Data,Length);
  int i, nh=0, nl=0;
  for(i=1; i<Length; i++) {
    if(Data[i] > m && Data[i] > Data[i-1]) // mind Data order: Data[0] is newest!
      nl++;
    else if(Data[i] < m && Data[i] < Data[i-1])
      nh++;
  }
  return 100.*(nl+nh)/(Length-1);}
```

This code is in C for Zorro, but meanwhile also versions for MT4, Amibroker, Ninja Trader, and other platforms have been programmed by users. As the name suggests, the indicator measures the meanness of the market – its tendency to revert to the mean after pretending to start a trend. If that happens too often, all trend following systems will bite the dust.

### The Three-Quarter Rule

Any series of independent random numbers reverts to the mean – or more precisely, to their median – with a probability of 75%. Assume you have a sequence of random, uncorrelated daily data – for instance, the daily rates of change of a random walk price curve. If Monday’s data value was above the median, then in 75% of all cases Tuesday’s data will be lower than Monday’s. And if Monday was below the median, 75% chance is that Tuesday will be higher. The proof of the 75% rule is relatively simple and won’t require integral calculus. Consider a data series with median **M**. By definition, half the values are less than **M** and half are greater (for simplicity’s sake we’re ignoring the case when a value is exactly **M**). Now combine the values to pairs each consisting of a value **Py** and the following value **Pt**. Thus each pair represents a change from **Py** to **Pt**. We now got a lot of changes that we divide into four sets:

1. **(Pt < M, Py < M)**
2. **(Pt < M, Py > M)**
3. **(Pt > M, Py < M)**
4. **(Pt > M, Py > M)**

These four sets have obviously the same number of elements – that is, 1/4 of all **Py->Pt** changes – when **Pt** and **Py** are uncorrelated, i.e. completely independent of one another. The value of **M** and the kind of data in the series won’t matter for this. Now how many data pairs revert to the median? All pairs that fulfill this condition: **(Py < M and Pt > Py) or (Py > M and Pt < Py)** The condition in the first bracket is fulfilled for half the data in set 1 (in the other half is **Pt** less than **Py**) and in the whole set 3 (because **Pt** is always higher than **Py** in set 3). So the first bracket is true for 1/2 * 1/4 + 1/4 = 3/8 of all data changes. Likewise, the second bracket is true in half the set 4 and in the whole set 2, thus also for 3/8 of all data changes. 3/8 + 3/8 yields 6/8, i.e. **75%**. This is the three-quarter rule for random numbers.

The **MMI** function just counts the number of data pairs for which the conditition is met, and returns their percentage. The **Data** series may contain prices or price changes. Prices have always some serial correlation: If EUR / USD today is at 1.20, it will also be tomorrow around 1.20. That it will end up tomorrow at 70 cents or 2 dollars per EUR is rather unlikely. This serial correlation is also true for a price series calculated from random numbers, as not the prices themselves are random, but their changes. Thus, the MMI function should return a smaller percentage, such as 55%, when fed with prices.

Unlike prices, price changes have not necessarily serial correlation. A one hundred percent efficient market has no correlation between the price change from yesterday to today and the price change from today to tomorrow. If the MMI function is fed with perfectly random price changes from a perfectly efficient market, it will return a value of about 75%. The less efficient and the more trending the market becomes, the more the MMI decreases. Thus a falling MMI is a indicator of an upcoming trend. A rising MMI hints that the market will get nastier, at least for trend trading systems.

### Using the MMI in a trend strategy

One could assume that MMI predicts the price direction. A high MMI value indicates a high chance of mean reversion, so when prices were moving up in the last time and MMI is high, can we expect a soon price drop? Unfortunately it doesn’t work this way. The probability of mean reversion is not evenly distributed over the **Length** of the **Data** interval. For the early prices it is high (since the median is computed from future prices), but for the late prices, at the very time when MMI is calculated, it is down to just 50%. Predicting the next price with the MMI would work as well as flipping a coin.

Another mistake would be using the MMI for detecting a cyclic or mean-reverting market regime. True, the MMI will rise in such a situation, but it will also rise when the market becomes more random and more effective. A rising MMI alone is no promise of profit by cycle trading systems.

So the MMI won’t tell us the next price, and it won’t tell us if the market is mean reverting or just plain mean, but it can reveal information about the success chance of trend following. For this we’re making an assumption: **Trend itself is trending**. The market does not jump in and out of trend mode suddenly, but with some inertia. Thus, when we know that MMI is rising, we assume that the market is becoming more efficient, more random, more cyclic, more reversing or whatever, but in any case bad for trend trading. However when MMI is falling, chances are good that the next beginning trend will last longer than normal.

This way the MMI can be an excellent trend filter – in theory. But we all know that there’s often a large gap between theory and practice, especially in algorithmic trading. So I’m now going to test what the Market Meanness Index does to the collection of the [900 trend following systems](http://www.financial-hacker.com/trend-and-exploiting-it/) that I’ve accumulated. For a first quick test, this was the equity curve of one of the systems, **TrendEMA**, without MMI (44% average annual return):

EMA strategy without MMI

![](https://financial-hacker.com/wp-content/uploads/2015/09/TrendEMA_EURUSD.png)

This is the same system with MMI (55% average annual return):

EMA strategy with MMI

![](https://financial-hacker.com/wp-content/uploads/2015/09/TrendEMA_EURUSD1.png)

We can see that the profit has doubled, from $250 to $500. The profit factor climbed from 1.2 to 1.8, and the number of trades (green and red lines) is noticeable reduced. On the other hand, the equity curve started with a drawdown that wasn’t there with the original system. So MMI obviously does not improve all trades. And this was just a randomly selected system. If our assumption about trend trendiness is true, the indicator should have a significant effect also on the other 899 systems.

This experiment will be the topic of the next article, in about a week. As usually I’ll include all the source code for anyone to reproduce it. Will the MMI miserably fail? Or improve only a few systems, but worsen others? Or will it light up the way to the Holy Grail of trend strategies? Let the market be the judge.

**Addendum (2022).** I thought my above proof of the 3/4 rule was trivial and no math required, but judging from some comments, it is not so. The rule appears a bit counter-intuitive at first glance. Some comments also confuse a random walk (for example, prices) with a random sequence (for example, price differences). Maybe I have just bad explained it – in that case, read up the proof anywhere else. I found the two most elaborate proofs of the 3/4 rule, geometrical and analytical, in this book:

(1) Andrew Pole, Statistical Arbitrage, Wiley 2007

And with no reading and no math: Take a dice, roll it 100 times, write down each result, get the median of all, and then count how often a pair reverted to the median.

Below you can see the MMI applied to a synthetic price curve, first sidewards (black) and then with added trend (blue). For using the MMI with real prices or price differences, read the next article.

![MMI](http://financial-hacker.com/images/Regime.png)

The code:

```
function run(){
	BarPeriod = 60;
	MaxBars = 1000;
	LookBack = 500;
	asset(""); // dummy asset
	ColorUp = ColorDn = 0; // don't plot a price curve
	set(PLOTNOW);

	if(Init) seed(1000);
	int TrendStart = LookBack + 0.4*(MaxBars-LookBack);
	vars Signals = series(1 + 0.5*genNoise());
	if(Bar > TrendStart)
		Signals[0] += 0.003*(Bar-TrendStart);
	int Color = RED;
	int TimePeriod = 0.5*LookBack;
	if(Bar < TrendStart)
		plot("Sidewards",Signals,MAIN,BLACK);
	else
		plot("Trending",Signals,MAIN,BLUE);
	plot("MMI",MMI(Signals,TimePeriod),NEW|LINE,Color);}
```