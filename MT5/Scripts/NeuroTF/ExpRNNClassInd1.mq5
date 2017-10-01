//+------------------------------------------------------------------+
//|                                                  ExpRNNClass.mq5 |
//|                        Copyright 2017, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
//--- input parameters
input int      ExtDepth=7;
input int      ExtDeviation=5;
input int      ExtBackstep=3;
input int      ATR=14;
input int      MACDFast=12;
input int      MACDSlow=29;
input int      K=5;
input int      D=3;
input int      Slow=3;
input int      NBars=50000;
input double   Split=0.98;
input double   norm=1;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
    int limit=Bars(Symbol(),Period());
    if(limit>NBars) limit=NBars;
    int h=iCustom(Symbol(),Period(),"Examples\\ZigZag",ExtDepth,ExtDeviation,ExtBackstep);
    double zh[],zl[],z[];
    ArrayResize(zh,limit);
    ArrayResize(zl,limit);
    ArrayResize(z,limit);
    CopyBuffer(h,1,0,limit,zh);
    CopyBuffer(h,2,0,limit,zl);
    CopyBuffer(h,0,0,limit,z);
    int h1=iATR(Symbol(),Period(),ATR);
    double at[];
    ArrayResize(at,limit);
    CopyBuffer(h1,0,0,limit,at);
    int h2=iMACD(Symbol(),Period(),MACDFast,MACDSlow,3,PRICE_MEDIAN);
    double macd[];
    ArrayResize(macd,limit);
    CopyBuffer(h2,0,0,limit,macd);
    int h3=iStochastic(Symbol(),Period(),K,D,Slow,MODE_SMA,STO_LOWHIGH);
    double rsi[];
    ArrayResize(rsi,limit);
    CopyBuffer(h3,0,0,limit,rsi);
    int tr=(int)(limit*Split);
    int ot=FileOpen("train.csv",FILE_WRITE|FILE_ANSI,";");
    string hdr="s1;s2;s3;s4;s5;s6;s7;s8;s9;s10;s11;s12;s13;s14;s15;s16;s17;s18;s19;s20;s21;s22;s23;s24;s25;s26;s27;s28;s29;s30;s31;s32;s33;s34;s35;s36;s37;s38;s39;s40;s41;s42;s43;s44;s45;l1;l2;l3";
    FileWrite(ot,hdr);
    for(int i=100;i<tr;i++)
    {
      string d="";
      for(int j=0;j<15;j++)
      {
         
         double delta=at[i-j]/norm;
         double delta1=macd[i-j]/norm;
         double delta2=rsi[i-j]/norm;
         d=d+DoubleToString(delta,15)+";"+DoubleToString(delta1,15)+";"+DoubleToString(delta2,15)+";";
      }
      string d1="0.0;0.0;1.0";
      if(z[i]>0)
      {
         if(zh[i]>0) d1="1.0;0.0;0.0";
         if(zl[i]>0) d1="0.0;1.0;0.0";
      }
      FileWrite(ot,d+d1);
    }
    FileClose(ot);
    ot=FileOpen("test.csv",FILE_WRITE|FILE_ANSI,";");
    FileWrite(ot,hdr);
    for(int i=tr;i<limit;i++)
    {
      string d="";
      for(int j=0;j<15;j++)
      {
         
         double delta=at[i-j]/norm;
         double delta1=macd[i-j]/norm;
         double delta2=rsi[i-j]/norm;
         d=d+DoubleToString(delta,15)+";"+DoubleToString(delta1,15)+";"+DoubleToString(delta2,15)+";";
      }
      string d1="0.0;0.0;1.0";
      if(z[i]>0)
      {
         if(zh[i]>0) d1="1.0;0.0;0.0";
         if(zl[i]>0) d1="0.0;1.0;0.0";
      }
      FileWrite(ot,d+d1);
    }
    FileClose(ot);
  }
//+------------------------------------------------------------------+
