import yfinance as yf
from tabulate import tabulate
import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob
import re


def get_entries_from_user():
    entries = []
    user_input = input("Enter a single entry, comma-separated entries, or a file path with entries: ").strip()

    if ',' in user_input:
        entries = [entry.strip() for entry in user_input.split(',')]
    elif '.' in user_input and '/' in user_input:
        try:
            with open(user_input, 'r') as file:
                entries = [line.strip() for line in file.readlines()]
        except FileNotFoundError:
            print(f"Error: The file '{user_input}' was not found.")
    else:
        entries.append(user_input)

    return entries


def format_number(value, is_currency=False):
    if isinstance(value, (int, float)):
        formatted_value = f"{value:,.2f}" if is_currency else f"{value:,}"
        if is_currency:
            formatted_value = f"${formatted_value}"
        return formatted_value
    return value


def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1], signal_line.iloc[-1]


def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def fetch_news_articles(api_key, stock):
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(q=f"{stock} stock", language='en', sort_by='relevancy', page_size=10)
    return all_articles['articles']


def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()


def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'


def fetch_data(entries, api_key):
    for entry in entries:
        try:
            stock = yf.Ticker(entry)
            info = stock.info

            company_name = info.get('longName', 'N/A')
            industry = info.get('industry', 'N/A')
            sector = info.get('sector', 'N/A')
            investor_site = info.get('website', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            prev_close = info.get('previousClose', 'N/A')
            open_price = info.get('open', 'N/A')
            day_low = info.get('dayLow', 'N/A')
            day_high = info.get('dayHigh', 'N/A')
            year_low = info.get('fiftyTwoWeekLow', 'N/A')
            year_high = info.get('fiftyTwoWeekHigh', 'N/A')
            year_average = info.get('fiftyDayAverage', 'N/A')
            earnings_quarterly = info.get('earningsQuarterlyGrowth', 'N/A')
            pe_ratio = info.get('pegRatio', 'N/A')
            revenue_growth = info.get('revenueGrowth', 'N/A')
            operating_cash = info.get('operatingCashflow', 'N/A')
            recommendation = info.get('recommendationKey', 'N/A')
            numb_analysts = info.get('numberOfAnalystOpinions', 'N/A')

            hist_data = stock.history(period="6mo")
            macd, signal_line = calculate_macd(hist_data)
            rsi = calculate_rsi(hist_data)

            prev_close_formatted = format_number(prev_close, is_currency=True)
            open_price_formatted = format_number(open_price, is_currency=True)
            day_low_formatted = format_number(day_low, is_currency=True)
            day_high_formatted = format_number(day_high, is_currency=True)
            year_low_formatted = format_number(year_low, is_currency=True)
            year_high_formatted = format_number(year_high, is_currency=True)
            year_average_formatted = format_number(year_average, is_currency=True)
            market_cap_formatted = format_number(market_cap, is_currency=True)
            operating_cash_formatted = format_number(operating_cash, is_currency=True)
            earnings_quarterly_formatted = format_number(earnings_quarterly)
            revenue_growth_formatted = format_number(revenue_growth)
            rsi_formatted = format_number(rsi)
            macd_formatted = format_number(macd)
            signal_formatted = format_number(signal_line)

            if isinstance(macd, (int, float)) and isinstance(signal_line, (int, float)):
                if macd < signal_line:
                    macd_formatted = color_text(macd_formatted + " (Bearish)", "31")
                elif macd > signal_line:
                    macd_formatted = color_text(macd_formatted + " (Bullish)", "32")
                else:
                    macd_formatted = macd_formatted + " (Neutral)"

            if isinstance(open_price, (int, float)) and isinstance(prev_close, (int, float)):
                if open_price < prev_close:
                    open_price_formatted = color_text(open_price_formatted, "31")
                else:
                    open_price_formatted = color_text(open_price_formatted, "32")

            if isinstance(earnings_quarterly, (int, float)):
                if earnings_quarterly < 0:
                    earnings_quarterly_formatted = color_text(earnings_quarterly_formatted, "31")
                else:
                    earnings_quarterly_formatted = color_text(earnings_quarterly_formatted, "32")

            if isinstance(revenue_growth, (int, float)):
                if revenue_growth < 0:
                    revenue_growth_formatted = color_text(revenue_growth_formatted, "31")
                else:
                    revenue_growth_formatted = color_text(revenue_growth_formatted, "32")

            if isinstance(rsi, (int, float)):
                if rsi < 30:
                    rsi_formatted = color_text(rsi_formatted + " (Oversold)", "32")
                elif rsi > 70:
                    rsi_formatted = color_text(rsi_formatted + " (Overbought)", "31")
                else:
                    rsi_formatted = rsi_formatted + " (Neutral)"

            stock_info_data = [
                ("Company Name", company_name),
                ("Industry", industry),
                ("Sector", sector),
                ("Investor Site", investor_site)
            ]

            performance_data = [
                ("Previous Close", prev_close_formatted),
                ("Open", open_price_formatted),
                ("Day Low", day_low_formatted),
                ("Day High", day_high_formatted),
                ("52 Week Low", year_low_formatted),
                ("52 Week High", year_high_formatted),
                ("50 Day Average", year_average_formatted)
            ]

            financial_data = [
                ("Market Cap", market_cap_formatted),
                ("Operating Cashflow", operating_cash_formatted),
                ("Earnings Quarterly Growth", earnings_quarterly_formatted),
                ("Revenue Growth", revenue_growth_formatted)
            ]

            technical_indicator = [
                ("MACD", macd_formatted),
                ("Signal Line", signal_formatted),
                ("RSI (14)", rsi_formatted)
            ]

            print(color_text(f"\nStock Information For {entry}:", 33))
            print(tabulate(stock_info_data, tablefmt="pretty", colalign=("left", "left")))

            print(color_text(f"\nStock Performance Overview:", 33))
            print(
                tabulate(performance_data, headers=["Metric", "Value"], tablefmt="pretty", colalign=("left", "right")))

            print(color_text(f"\nFinancial Overview:", 33))
            print(tabulate(financial_data, headers=["Metric", "Value"], tablefmt="pretty", colalign=("left", "right")))

            print(color_text(f"\nTechnical Indicators:", 33))
            print(tabulate(technical_indicator, headers=["Metric", "Value"], tablefmt="pretty",
                           colalign=("left", "right")))

            # Fetch and analyze news articles
            news_articles = fetch_news_articles(api_key, entry)
            sentiments = [analyze_sentiment(preprocess_text(article['description'])) for article in news_articles if
                          article['description']]

            if sentiments:
                positive_count = sentiments.count('positive')
                negative_count = sentiments.count('negative')
                neutral_count = sentiments.count('neutral')
                total = len(sentiments)

                if positive_count > negative_count and positive_count > neutral_count:
                    overall_sentiment = color_text("Positive", "32")
                elif negative_count > positive_count and negative_count > neutral_count:
                    overall_sentiment = color_text("Negative", "31")
                else:
                    overall_sentiment = color_text("Neutral", "33")

                positive_count_colored = color_text(str(positive_count), "32")
                negative_count_colored = color_text(str(negative_count), "31")

                print(color_text(f"\nOverall Sentiment from Recent News:", 33))
                print(
                    f"Sentiment: {overall_sentiment} (Negative: {negative_count_colored}, Neutral: {neutral_count}, Positive: {positive_count_colored})")
            else:
                print(color_text("\nNo recent news articles found.", 33))

        except Exception as e:
            print(color_text(f"Error fetching data for {entry}: {e}", 31))

def main():
    banner = r"""
   _____    __                      __               _    __
  / ___/   / /_   ____     _____   / /__   ____     (_)  / /  ___     _____      ____     __  __
  \__ \   / __/  / __ \   / ___/  / //_/  / __ \   / /  / /  / _ \   / ___/     / __ \   / / / /
 ___/ /  / /_   / /_/ /  / /__   / ,<    / /_/ /  / /  / /  /  __/  / /    _   / /_/ /  / /_/ /
/____/   \__/   \____/   \___/  /_/|_|  / .___/  /_/  /_/   \___/  /_/    (_) / .___/   \__, /
                                       /_/                                   /_/       /____/
"""
    print(color_text(banner, 33))

    api_key = "3c1234a6ccf54800af0432c5bd6fe859"  # Replace with your actual API key
    entries = get_entries_from_user()
    if entries:
        fetch_data(entries, api_key)
    else:
        print("No valid entries provided.")


if __name__ == "__main__":
    main()
