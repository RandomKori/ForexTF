//+------------------------------------------------------------------+
//|                                                       ResNet.mq5 |
//|                        Copyright 2017, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 1
#property indicator_plots   1
//--- plot Label1
#property indicator_label1  "Label1"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrYellow
#property indicator_style1  STYLE_SOLID
#property indicator_width1  4
//--- input parameters
input string   path="D:\\ModelFORTS\\DenseClass";
input int      NBars=1000;
input double   norm=1.0;
input double   Signal=0.5;
input double   NoSignal=0.5;

#import "NewNeuroDense.dll"
void LoadModel(string s);
void EvalModel(double &inp[], double &out[]);
void DeInit();
#import

//--- indicator buffers
double         Label1Buffer[];
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0,Label1Buffer,INDICATOR_DATA);
//--- setting a code from the Wingdings charset as the property of PLOT_ARROW
   PlotIndexSetInteger(0,PLOT_ARROW,159);
   PlotIndexSetDouble(0,PLOT_EMPTY_VALUE,0.0);
   LoadModel(path);
//---
   return(INIT_SUCCEEDED);
  }
  
  void OnDeinit(const int reason)
  {
   DeInit();
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//---
   double in[45],ot[2];
   int limit;
   if(prev_calculated==0)
      limit=0;
   else limit=prev_calculated-1;
   for(int i=limit;i<rates_total && !IsStopped();i++)
   {
      int index=0;
      if(i<Bars(Symbol(),Period())-NBars-1) continue;
      for(int j=1;j<16;j++)
      {
         double delta=(high[i-j]-low[i-j])/norm;
         double delta1=(high[i-j]-high[i-j-1])/norm;
         double delta2=(low[i-j]-low[i-j-1])/norm;
         in[index]=delta;
         in[index+1]=delta1;
         in[index+2]=delta2;
         index=index+3;
      }
      EvalModel(in,ot);
      Print(ot[0]," ",ot[1]);
      Label1Buffer[i-1]=0.0;
      if(ot[0]>Signal && ot[1]<NoSignal) Label1Buffer[i-1]=high[i-1];
      if(ot[0]<NoSignal && ot[1]>Signal) Label1Buffer[i-1]=low[i-1];
   }
//--- return value of prev_calculated for next call
   return(rates_total);
  }
//+------------------------------------------------------------------+
