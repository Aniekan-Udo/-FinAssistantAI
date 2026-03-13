from datetime import date
import yfinance as yf

class StockDataFetcher():
    def __init__(self, ticker):
        self.ticker= ticker
        self._ticker_obj = None

    @property
    def ticker_obj(self):
        """Cached ticker object"""
        if not self._ticker_obj:
            self._ticker_obj = yf.Ticker(self.ticker)
        return self._ticker_obj
    
    def _company_information(self) -> dict:
        return self.ticker_obj.get_info()
        

    def _last_dividend_and_earnings_date(self) -> dict:
        return self.ticker_obj.get_calendar()

    def _summary_of_mutual_fund_holders(self) -> dict:
        mf_holders = self.ticker_obj.get_mutualfund_holders()
        return mf_holders.to_dict(orient="records")

    def _summary_of_institutional_holders(self) -> dict:
        inst_holders = self.ticker_obj.get_institutional_holders()
        return inst_holders.to_dict(orient="records")

    def _stock_grade_upgrades_downgrades(self) -> dict:
        curr_year = date.today().year
        upgrades_downgrades = self.ticker_obj.get_upgrades_downgrades()
        if upgrades_downgrades is not None:
            upgrades_downgrades = upgrades_downgrades.loc[upgrades_downgrades.index > f"{curr_year}-01-01"]
            upgrades_downgrades = upgrades_downgrades[upgrades_downgrades["Action"].isin(["up", "down"])]
        return upgrades_downgrades.to_dict(orient="records") if upgrades_downgrades is not None else {}

    def _stock_splits_history(self) -> dict:
        hist_splits = self.ticker_obj.get_splits()
        return hist_splits.to_dict()

    def _stock_news(self) -> dict:
        return self.ticker_obj.get_news()

    def _stock_info(self) -> dict:
        return self.ticker_obj.get_analyst_price_targets()

    def _stock_history(self, period="1mo") -> dict:
        """Fetch historical price data for charting."""
        hist = self.ticker_obj.history(period=period)
        if hist.empty:
            return {}
        
        # Reset index to get Date as a column
        hist = hist.reset_index()
        
        # Format for Chart.js: labels (dates) and data (closing prices)
        return {
            "labels": hist['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "prices": hist['Close'].round(2).tolist()
        }
