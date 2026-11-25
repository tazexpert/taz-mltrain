//+------------------------------------------------------------------+
//|                                     XAUUSD_M5_MultiSetup_EA.mq4  |
//|                    Multi-Setup (InsideBar + MACD + EMA9/20) EA   |
//|                    2-Order Execution (TP1/TP2 + Move to BE)      |
//+------------------------------------------------------------------+
#property copyright "Nachapat Chaiyakhum"
#property version   "1.00"
#property strict

//--- Input Parameters
input string   Sec1           = "=== Strategy & Setup Filters ===";
input bool     UseInsideBar   = true;   // Enable S6: Inside Bar Breakout
input bool     UseMACD        = true;   // Enable S2: MACD Zero-Line Pullback Cross
input bool     UseEMA920      = true;   // Enable S3: EMA 9/20 Crossover
input double   MinM5ADX       = 18.0;   // Minimum M5 ADX(14)

input string   Sec2           = "=== Risk & 2-Order Management ===";
input double   RiskPctPerOrd  = 0.25;   // Risk % per SINGLE order (2 orders = 2x risk)
input double   SL_ATR_Mult    = 1.5;    // SL distance = M5 ATR(14) * Multiplier
input double   RR1            = 1.0;    // Order 1 TP Risk-Reward Ratio
input double   RR2            = 2.5;    // Order 2 TP Risk-Reward Ratio (> RR1)
input bool     MoveBE_OnTP1   = true;   // Move Order 2 SL to Breakeven at TP1
input double   MaxSpreadPts   = 40.0;   // Max allowed spread in points
input int      MagicNumber    = 202611; // Unique EA Magic Number
input int      Slippage       = 20;     // Max slippage in points

//--- Global Variables
datetime g_lastM5BarTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   if(Period() != PERIOD_M5)
      Print("Warning: Attach this EA to the M5 timeframe for optimal execution.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1. Manage Breakeven for Order 2 on every tick
   if(MoveBE_OnTP1)
      ManageBreakeven();

   // 2. Check for new closed M5 bar
   datetime currentM5Time = iTime(Symbol(), PERIOD_M5, 0);
   if(currentM5Time == g_lastM5BarTime)
      return;
   g_lastM5BarTime = currentM5Time;

   // 3. Enforce non-overlapping rule: only 1 active 2-order set at a time
   if(CountActiveOrders() > 0)
      return;

   // 4. Spread filter
   double currentSpread = MarketInfo(Symbol(), MODE_SPREAD);
   if(currentSpread > MaxSpreadPts)
      return;

   // 5. Evaluate Higher Timeframe (H1) Trend & M5 ADX (using closed bar shift=1)
   double h1_close  = iClose(Symbol(), PERIOD_H1, 1);
   double h1_ema50  = iMA(Symbol(), PERIOD_H1, 50, 0, MODE_EMA, PRICE_CLOSE, 1);
   double h1_ema200 = iMA(Symbol(), PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE, 1);

   bool h1_up = (h1_close > h1_ema50) && (h1_ema50 > h1_ema200);
   bool h1_dn = (h1_close < h1_ema50) && (h1_ema50 < h1_ema200);

   if(!h1_up && !h1_dn)
      return;

   double m5_adx = iADX(Symbol(), PERIOD_M5, 14, PRICE_CLOSE, MODE_MAIN, 1);
   if(m5_adx < MinM5ADX)
      return;

   // 6. Check 3 Base Setups on closed M5 bar (shift=1)
   bool sigBuy  = false;
   bool sigSell = false;
   string tag   = "";

   //--- S6: Inside Bar Breakout (shift=2 is inside bar, shift=3 is mother bar)
   if(UseInsideBar)
   {
      double h1 = iHigh(Symbol(), PERIOD_M5, 1);
      double l1 = iLow(Symbol(), PERIOD_M5, 1);
      double c1 = iClose(Symbol(), PERIOD_M5, 1);
      double h2 = iHigh(Symbol(), PERIOD_M5, 2);
      double l2 = iLow(Symbol(), PERIOD_M5, 2);
      double h3 = iHigh(Symbol(), PERIOD_M5, 3);
      double l3 = iLow(Symbol(), PERIOD_M5, 3);

      bool isIB = (h2 < h3) && (l2 > l3);
      if(isIB)
      {
         if(h1_up && c1 > h3) { sigBuy  = true; tag = "IB"; }
         if(h1_dn && c1 < l3) { sigSell = true; tag = "IB"; }
      }
   }

   //--- S2: MACD Zero-Line Pullback Cross
   if(UseMACD && !sigBuy && !sigSell)
   {
      double macd1 = iMACD(Symbol(), PERIOD_M5, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 1);
      double sig1  = iMACD(Symbol(), PERIOD_M5, 12, 26, 9, PRICE_CLOSE, MODE_SIGNAL, 1);
      double macd2 = iMACD(Symbol(), PERIOD_M5, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 2);
      double sig2  = iMACD(Symbol(), PERIOD_M5, 12, 26, 9, PRICE_CLOSE, MODE_SIGNAL, 2);

      if(h1_up && macd1 > sig1 && macd2 <= sig2 && macd1 < 0) { sigBuy  = true; tag = "MACD"; }
      if(h1_dn && macd1 < sig1 && macd2 >= sig2 && macd1 > 0) { sigSell = true; tag = "MACD"; }
   }

   //--- S3: EMA 9/20 Crossover
   if(UseEMA920 && !sigBuy && !sigSell)
   {
      double ema9_1  = iMA(Symbol(), PERIOD_M5, 9, 0, MODE_EMA, PRICE_CLOSE, 1);
      double ema20_1 = iMA(Symbol(), PERIOD_M5, 20, 0, MODE_EMA, PRICE_CLOSE, 1);
      double ema9_2  = iMA(Symbol(), PERIOD_M5, 9, 0, MODE_EMA, PRICE_CLOSE, 2);
      double ema20_2 = iMA(Symbol(), PERIOD_M5, 20, 0, MODE_EMA, PRICE_CLOSE, 2);

      if(h1_up && ema9_1 > ema20_1 && ema9_2 <= ema20_2) { sigBuy  = true; tag = "EMA"; }
      if(h1_dn && ema9_1 < ema20_1 && ema9_2 >= ema20_2) { sigSell = true; tag = "EMA"; }
   }

   // 7. Execute 2-Order Set
   if(sigBuy)
      OpenTwoOrderSet(OP_BUY, tag);
   else if(sigSell)
      OpenTwoOrderSet(OP_SELL, tag);
}

//+------------------------------------------------------------------+
//| Open 2-Order Set (Order 1 = TP1, Order 2 = TP2)                  |
//+------------------------------------------------------------------+
void OpenTwoOrderSet(int cmd, string tag)
{
   double atr = iATR(Symbol(), PERIOD_M5, 14, 1);
   double slDist = atr * SL_ATR_Mult;
   if(slDist <= 0) return;

   double lots = CalculateLotSize(slDist);
   if(lots <= 0) return;

   RefreshRates();
   double price = (cmd == OP_BUY) ? Ask : Bid;
   double sl    = (cmd == OP_BUY) ? (price - slDist) : (price + slDist);
   double tp1   = (cmd == OP_BUY) ? (price + slDist * RR1) : (price - slDist * RR1);
   double tp2   = (cmd == OP_BUY) ? (price + slDist * RR2) : (price - slDist * RR2);

   sl  = NormalizeDouble(sl, Digits);
   tp1 = NormalizeDouble(tp1, Digits);
   tp2 = NormalizeDouble(tp2, Digits);

   // Open Order 1 (TP1)
   int t1 = OrderSend(Symbol(), cmd, lots, price, Slippage, sl, tp1, tag + "_TP1", MagicNumber, 0, clrDodgerBlue);
   if(t1 < 0) Print("OrderSend TP1 failed. Error: ", GetLastError());

   // Open Order 2 (TP2)
   int t2 = OrderSend(Symbol(), cmd, lots, price, Slippage, sl, tp2, tag + "_TP2", MagicNumber, 0, clrLimeGreen);
   if(t2 < 0) Print("OrderSend TP2 failed. Error: ", GetLastError());
}

//+------------------------------------------------------------------+
//| Move Order 2 SL to Breakeven once price reaches TP1 distance     |
//+------------------------------------------------------------------+
void ManageBreakeven()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != Symbol() || OrderMagicNumber() != MagicNumber) continue;
      if(StringFind(OrderComment(), "_TP2") < 0) continue;

      double openPrice = OrderOpenPrice();
      double curSL     = OrderStopLoss();
      double curTP     = OrderTakeProfit();

      // Recover original 1R distance from TP2 distance (TP2_dist / RR2)
      double slDist = MathAbs(curTP - openPrice) / RR2;
      double tp1TriggerDist = slDist * RR1;

      RefreshRates();
      if(OrderType() == OP_BUY)
      {
         if(Bid >= openPrice + tp1TriggerDist && curSL < openPrice)
         {
            double newSL = NormalizeDouble(openPrice, Digits);
            if(!OrderModify(OrderTicket(), openPrice, newSL, curTP, 0, clrYellow))
               Print("OrderModify BE BUY failed. Error: ", GetLastError());
         }
      }
      else if(OrderType() == OP_SELL)
      {
         if(Ask <= openPrice - tp1TriggerDist && (curSL > openPrice || curSL == 0))
         {
            double newSL = NormalizeDouble(openPrice, Digits);
            if(!OrderModify(OrderTicket(), openPrice, newSL, curTP, 0, clrYellow))
               Print("OrderModify BE SELL failed. Error: ", GetLastError());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate Lot Size based on Risk % and SL Distance               |
//+------------------------------------------------------------------+
double CalculateLotSize(double slDist)
{
   double riskMoney = AccountBalance() * (RiskPctPerOrd / 100.0);
   double tickValue = MarketInfo(Symbol(), MODE_TICKVALUE);
   double tickSize  = MarketInfo(Symbol(), MODE_TICKSIZE);
   if(tickValue <= 0 || tickSize <= 0) return(0.0);

   double slTicks = slDist / tickSize;
   double rawLots = riskMoney / (slTicks * tickValue);

   double minLot  = MarketInfo(Symbol(), MODE_MINLOT);
   double maxLot  = MarketInfo(Symbol(), MODE_MAXLOT);
   double lotStep = MarketInfo(Symbol(), MODE_LOTSTEP);

   double lots = MathFloor(rawLots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| Count currently open orders for this EA                          |
//+------------------------------------------------------------------+
int CountActiveOrders()
{
   int count = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() == Symbol() && OrderMagicNumber() == MagicNumber)
         count++;
   }
   return count;
}
//+------------------------------------------------------------------+