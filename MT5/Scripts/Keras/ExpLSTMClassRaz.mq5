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
input int      NBars=50000;
input double   Split=0.98;
input double   norm=1.0;
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
    string hdr="s1;s2;s3;s4;s5;s6;s7;s8;s9;s10;s11;s12;s13;s14;s15;s16;s17;s18;s19;s20;s21;s22;s23;s24;s25;s26;s27;s28;s29;s30;l1;l2;l3";
    FileWrite(ot,hdr);
    for(int i=11;i<tr;i++)
    {
      string d="";
      for(int j=0;j<10;j++)
      {
         
         double delta=(his[i-j].high-his[i-j].low)/norm;
         double delta1=(his[i-j].high-his[i-j-1].high)/norm;
         double delta2=(his[i-j].low-his[i-j-1].low)/norm;
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
      for(int j=0;j<10;j++)
      {
         
         double delta=(his[i-j].high-his[i-j].low)/norm;
         double delta1=(his[i-j].high-his[i-j-1].high)/norm;
         double delta2=(his[i-j].low-his[i-j-1].low)/norm;
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
