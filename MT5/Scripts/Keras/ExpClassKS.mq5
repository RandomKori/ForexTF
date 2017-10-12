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

input int      NBars=30000;
input double   Split=0.98;
input double   norm=1.0;
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
    string hdr="s1;s2;s3;s4;s5;s6;s7;s8;s9;s10;s11;s12;s13;s14;s15;s16;s17;s18;s19;s20;s21;s22;s23;s24;s25;s26;s27;s28;s29;s30;s31;s32;s33;s34;s35;s36;s37;s38;s39;s40;s41;s42;s43;s44;s45;l1";
    FileWrite(ot,hdr);
    for(int i=17;i<tr;i++)
    {
      string d="";
      for(int j=1;j<16;j++)
      {
         
         double delta=MathLog(his[i-j].high/his[i-j].low)/norm;
         double delta1=MathLog(his[i-j].high/his[i-j-1].high)/norm;
         double delta2=MathLog(his[i-j].low/his[i-j-1].low)/norm;
         d=d+DoubleToString(delta,15)+";"+DoubleToString(delta1,15)+";"+DoubleToString(delta2,15)+";";
      }
      string d1="";
      if(his[i].high>his[i-1].high &&his[i].low>his[i-1].low) d1="0";
      if(his[i].high<his[i-1].high &&his[i].low<his[i-1].low) d1="1";
      if(his[i].high>his[i-1].high &&his[i].low<his[i-1].low) d1="2";
      if(his[i].high<his[i-1].high &&his[i].low>his[i-1].low) d1="3";
      if(his[i].high==his[i-1].high &&his[i].low==his[i-1].low) d1="4";
      if(his[i].high==his[i-1].high &&his[i].low>his[i-1].low) d1="5";
      if(his[i].high==his[i-1].high &&his[i].low<his[i-1].low) d1="6";
      if(his[i].high>his[i-1].high &&his[i].low==his[i-1].low) d1="7";
      if(his[i].high<his[i-1].high &&his[i].low==his[i-1].low) d1="8";
      FileWrite(ot,d+d1);
    }
    FileClose(ot);
    ot=FileOpen("test.csv",FILE_WRITE|FILE_ANSI,";");
    FileWrite(ot,hdr);
    for(int i=tr;i<limit;i++)
    {
      string d="";
      for(int j=1;j<16;j++)
      {
         
         double delta=MathLog(his[i-j].high/his[i-j].low)/norm;
         double delta1=MathLog(his[i-j].high/his[i-j-1].high)/norm;
         double delta2=MathLog(his[i-j].low/his[i-j-1].low)/norm;
         d=d+DoubleToString(delta,15)+";"+DoubleToString(delta1,15)+";"+DoubleToString(delta2,15)+";";
      }
      string d1="";
      if(his[i].high>his[i-1].high &&his[i].low>his[i-1].low) d1="0";
      if(his[i].high<his[i-1].high &&his[i].low<his[i-1].low) d1="1";
      if(his[i].high>his[i-1].high &&his[i].low<his[i-1].low) d1="2";
      if(his[i].high<his[i-1].high &&his[i].low>his[i-1].low) d1="3";
      if(his[i].high==his[i-1].high &&his[i].low==his[i-1].low) d1="4";
      if(his[i].high==his[i-1].high &&his[i].low>his[i-1].low) d1="5";
      if(his[i].high==his[i-1].high &&his[i].low<his[i-1].low) d1="6";
      if(his[i].high>his[i-1].high &&his[i].low==his[i-1].low) d1="7";
      if(his[i].high<his[i-1].high &&his[i].low==his[i-1].low) d1="8";
      FileWrite(ot,d+d1);
    }
    FileClose(ot);
  }
//+------------------------------------------------------------------+
