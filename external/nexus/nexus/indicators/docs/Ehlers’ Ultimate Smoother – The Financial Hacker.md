# Ehlers’ Ultimate Smoother – The Financial Hacker

[https://financial-hacker.com/ehlers-ultimate-smoother/#more-4717](https://financial-hacker.com/ehlers-ultimate-smoother/#more-4717)

![](https://financial-hacker.com/wp-content/uploads/2024/05/hacker1.jpg)

*In TASC 3/24, John Ehlers presented several functions for smoothing a price curve without lag, smoothing it even more, and applying a highpass and bandpass filter. No-lag smoothing, highpass, and bandpass filters are already available in the indicator library of the Zorro platform, but not Ehlers’ latest invention, the Ultimate Smoother. It achieves its tremendous smoothing power by subtracting the high frequency components from the price curve, using a highpass filter.*

The function below is a straightforward conversion of Ehlers’ EasyLanguage code to C:

```
var UltimateSmoother (var *Data, int Length){
  var f = (1.414*PI) / Length;
  var a1 = exp(-f);
  var c2 = 2*a1*cos(f);
  var c3 = -a1*a1;
  var c1 = (1+c2-c3)/4;
  vars US = series(*Data,4);
  return US[0] = (1-c1)*Data[0] + (2*c1-c2)*Data[1] - (c1+c3)*Data[2]+ c2*US[1] + c3*US[2];}
```

For comparing lag and smoothing power, we apply the ultimate smoother, the super smoother from Zorro’s indicator library, and a standard EMA to an ES chart from 2023:

```
void run(){
  BarPeriod = 1440;
  StartDate = 20230201;
  EndDate = 20231201;
  assetAdd("ES","YAHOO:ES=F");
  asset("ES");
  int Length = 30;
  plot("UltSmooth", UltimateSmoother(seriesC(),Length),LINE,MAGENTA);
  plot("Smooth",Smooth(seriesC(),Length),LINE,RED);
  plot("EMA",EMA(seriesC(),3./Length),LINE,BLUE);}}
```

The resulting chart replicates the ES chart in the article. The EMA is shown in blue, the super smoothing filter in red, and the ultimate smoother in magenta:

![](https://financial-hacker.com/wp-content/uploads/2024/05/word-image-4717-1.png)

We can see that the ultimate smoother produces indeed the best, albeit smoothed, representation of the price curve.

In TASC 4/24, Ehlers also presented two band indicators based on his Ultimate Smoother. Band indicators can be used to trigger long or short positions when the price hits the upper or lower band. The first band indicator, the **Ultimate Channel**, is again a straightforward conversion to the C language from Ehlers’ TradeStation code:

```
var UltimateChannel(int Length,int STRLength,int NumSTRs){
  var TH = max(priceC(1),priceH());
  var TL = min(priceC(1),priceL());
  var STR = UltimateSmoother(series(TH-TL),STRLength);
  var Center = UltimateSmoother(seriesC(),Length);
  rRealUpperBand = Center + NumSTRs*STR;
  rRealLowerBand = Center - NumSTRs*STR;
  return Center;}
```

**rRealUpperBand** and **rRealLowerBand** are pre-defined global variables that are used by band indicators in the indicator library of the Zorro platform. For testing the new indicator, we apply it to an ES chart:

```
void run(){
  BarPeriod = 1440;
  StartDate = 20230301;
  EndDate = 20240201;
  assetAdd("ES","YAHOO:ES=F");
  asset("ES");
  UltimateChannel(20,20,1);
  plot("UltChannel1",rRealUpperBand,BAND1,BLUE);
  plot("UltChannel2",rRealLowerBand,BAND2,BLUE|TRANSP);}
```

The resulting chart replicates the ES chart in Ehlers’ article:

![](https://financial-hacker.com/wp-content/uploads/2024/05/word-image-4717-1-1.png)

The second band indicator, **Ultimate Bands**, requires less code than Ehlers’ implementation, since lite-C can apply functions to a whole data series:

```
var UltimateBands(int Length,int NumSDs){
  var Center = UltimateSmoother(seriesC(),Length);
  vars Diffs = series(priceC()-Center);
  var SD = sqrt(SumSq(Diffs,Length)/Length);
  rRealUpperBand = Center + NumSDs*SD;
  rRealLowerBand = Center - NumSDs*SD; return Center;}
```

Again applied to the ES chart:

```
void run(){
  BarPeriod = 1440;
  StartDate = 20230301;
  EndDate = 20240201;
  assetAdd("ES","YAHOO:ES=F");
  asset("ES");
  UltimateBands(20,1);
  plot("UltBands1",rRealUpperBand,BAND1,BLUE);
  plot("UltBands2",rRealLowerBand,BAND2,BLUE|TRANSP);}
```

![](https://financial-hacker.com/wp-content/uploads/2024/05/word-image-4717-2.png)

We can see that both indicators produce relatively similar bands with low lag. The code of the Ultimate Smoother and the bands can be downloaded from the 2024 script repository.