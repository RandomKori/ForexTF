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
    
    MqlRates his[];
    ArrayResize(his,limit);
    CopyRates(Symbol(),Period(),0,limit,his);
    int tr=(int)(limit*Split);
    int ot=FileOpen("train.csv",FILE_WRITE|FILE_ANSI,";");
    string hdr="s1;s2;s3;s4;s5;s6;s7;s8;s9;s10;l1";
    FileWrite(ot,hdr);
    for(int i=12;i<tr;i++)
    {
      string d="";
      for(int j=1;j<11;j++)
      {
         
         
         double delta1=(his[i-j].high/his[i-j-1].high)/norm;
         
         d=d+DoubleToString(delta1,15)+";";
      }
      string d1="";
      double l1=(his[i].high/his[i-1].high)/norm;
      
      d1=DoubleToString(l1,15);
      FileWrite(ot,d+d1);
    }
    FileClose(ot);
    ot=FileOpen("test.csv",FILE_WRITE|FILE_ANSI,";");
    FileWrite(ot,hdr);
    for(int i=tr;i<limit;i++)
    {
      string d="";
      for(int j=1;j<11;j++)
      {
         
         double delta1=(his[i-j].high/his[i-j-1].high)/norm;
         
         d=d+DoubleToString(delta1,15)+";";
      }
      string d1="";
       double l1=(his[i].high/his[i-1].high)/norm;
      
      d1=DoubleToString(l1,15);
      FileWrite(ot,d+d1);
    }
    FileClose(ot);
  }
//+------------------------------------------------------------------+
