//+------------------------------------------------------------------+
//|                                         ExpResNetMedianPrice.mq5 |
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
input int      NBars=50000;
input double   Split=0.98;
input double   norm=10.0;
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
    MqlRates his[];
    ArrayResize(his,limit);
    CopyRates(Symbol(),Period(),0,limit,his);
    int tr=(int)(limit*Split);
    int ot=FileOpen("train.csv",FILE_WRITE|FILE_ANSI,";");
    
    for(int i=61;i<tr;i++)
    {
      string d="";
      for(int j=0;j<60;j++)
      {
         
         double delta=((his[i-j].high+his[i-j].low)/2)-((his[i-j-1].high+his[i-j-1].low)/2)/norm;
         d=d+DoubleToString(delta,15)+";";
      }
      string d1="3";
      if(z[i]>0)
      {
         if(zh[i]>0) d1="1";
         if(zl[i]>0) d1="2";
      }
      FileWrite(ot,d+d1);
    }
    FileClose(ot);
    ot=FileOpen("test.csv",FILE_WRITE|FILE_ANSI,";");
    
    for(int i=tr;i<limit;i++)
    {
      string d="";
      for(int j=0;j<60;j++)
      {
         
         double delta=((his[i-j].high+his[i-j].low)/2)-((his[i-j-1].high+his[i-j-1].low)/2)/norm;
         d=d+DoubleToString(delta,15)+";";
      }
      string d1="3";
      if(z[i]>0)
      {
         if(zh[i]>0) d1="1";
         if(zl[i]>0) d1="2";
      }
      FileWrite(ot,d+d1);
    }
    FileClose(ot);
  }
//+------------------------------------------------------------------+
