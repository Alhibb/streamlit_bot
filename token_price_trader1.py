import asyncio
import aiohttp
import argparse
import logging
import os
from typing import Dict, Optional, Tuple
import streamlit as st
from streamlit_extras.card import card
import plotly.express as px
from solders.pubkey import Pubkey
from agentipy.agent import SolanaAgentKit
from agentipy.tools.get_token_data import TokenDataManager
from agentipy.tools.trade import TradeManager
from agentipy.tools.get_balance import BalanceFetcher
from spl.token.async_client import AsyncToken
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import create_associated_token_account

# Known Solana Mint Addresses
SOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")
USDC_MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
WBTC_MINT = Pubkey.from_string("3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh")
JUP_MINT = Pubkey.from_string("JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN")

# API Endpoints
JUP_API = "https://quote-api.jup.ag/v6"
RAYDIUM_API = "https://api.raydium.io/v2/main/price"
DEXSCREENER_API = "https://api.dexscreener.com/latest/dex/tokens/"

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Command-Line Arguments 
parser = argparse.ArgumentParser(description="Solana Token Price Trader")
parser.add_argument("--slippage", type=int, default=50, help="Slippage tolerance in basis points")
parser.add_argument("--probe-amount", type=float, default=0.1, help="Probe amount in SOL")
parser.add_argument("--rpc-url", type=str, default="https://api.mainnet-beta.solana.com", help="Solana RPC URL")
args = parser.parse_args()

SLIPPAGE_BPS = args.slippage
PROBE_AMOUNT = int(args.probe_amount * 10**9)
PROBE_AMOUNT_JUP = 1 * 10**6
PRICE_DECIMALS = ".7f"
BALANCE_DECIMALS = ".7f"

# Initialize SolanaAgentKit
agent = SolanaAgentKit(
    private_key=os.getenv("SOLANA_PRIVATE_KEY", ""),
    rpc_url=args.rpc_url
)

# Set page configuration for Streamlit
st.set_page_config(
    page_title="Solana Token Price Trader",
    page_icon=":solar-panel:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Font Awesome and Roboto via CDN
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
    .stApp {
        background-color: #2E7D32; /* Darker green background */
        color: #FFFF00; /* Yellow text (adjust to #FFD700 for gold if too bright) */
        font-family: 'Roboto', sans-serif;
    }
    .stCard {
        background-color: #9CCC65 !important; /* Light yellow-green for cards */
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        padding: 15px !important;
    }
    .stButton>button {
        background-color: #4CAF50 !important; /* Green for buttons */
        color: #FFFF00 !important; /* Yellow text (adjust to #FFD700 if needed) */
        border-radius: 8px !important;
        padding: 10px 20px !important;
        transition: background-color 0.3s !important;
    }
    .stButton>button:hover {
        background-color: #388E3C !important; /* Darker green on hover */
    }
    .stTextInput>div>input, .stSelectbox>div>select {
        background-color: #9CCC65 !important; /* Light yellow-green for inputs */
        color: #FFFF00 !important; 
        border: 1px solid #4CAF50 !important; /* Green border */
        border-radius: 8px !important;
    }
    .fa-icon {
        color: #4CAF50; /* Green for icons */
        margin-right: 8px;
    }
    .button-with-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state for storing data
if "prices" not in st.session_state:
    st.session_state.prices = {}
if "token_mint" not in st.session_state:
    st.session_state.token_mint = None
if "sol_balance" not in st.session_state:
    st.session_state.sol_balance = 0.0
if "token_balance" not in st.session_state:
    st.session_state.token_balance = 0.0
if "price_history" not in st.session_state:
    st.session_state.price_history = {}

async def main():
    global SLIPPAGE_BPS, PROBE_AMOUNT  

    # Sidebar for settings
    st.sidebar.markdown("### Settings <i class='fas fa-cog fa-lg' style='color: #4CAF50;'></i>", unsafe_allow_html=True)
    slippage = st.sidebar.slider("Slippage (bps)", 0, 500, SLIPPAGE_BPS, key="slippage")
    probe_amount = st.sidebar.number_input("Probe Amount (SOL)", 0.01, 1.0, 0.1, key="probe_amount")
    rpc_url = st.sidebar.text_input("RPC URL", args.rpc_url, key="rpc_url")

    # Update global constants based on sidebar
    SLIPPAGE_BPS = slippage
    PROBE_AMOUNT = int(probe_amount * 10**9)

    # Main content with card layout
    st.markdown("# Solana Token Price Trader <i class='fas fa-solar-panel fa-2x' style='color: #4CAF50;'></i>", unsafe_allow_html=True)

    # Token Input Card
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            ticker = st.text_input("Enter Token Ticker", value="SOL", key="ticker_input", help="Enter SOL, USDC, BTC, JUP, or any Solana token ticker")
        with col2:
            refresh_button = st.button("Refresh Prices", key="refresh", help="Update token prices and balances")
            st.markdown(f'<div class="button-with-icon"><i class="fas fa-sync-alt fa-lg"></i></div>', unsafe_allow_html=True)
            if refresh_button:
                await update_prices_and_balances(ticker)

    # Price and Balance Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Prices <i class='fas fa-chart-line fa-lg' style='color: #4CAF50;'></i>", unsafe_allow_html=True)
        for dex, price in st.session_state.prices.items():
            card(
                title=dex,
                text=f"{price:{PRICE_DECIMALS}} SOL/token",
                key=f"price_card_{dex}"
            )
            st.markdown(f'<i class="fas fa-chart-pie fa-lg fa-icon"></i>', unsafe_allow_html=True)  
        # Price History Chart 
        if st.session_state.price_history:
            price_data = [{"Time": i, "Price": p, "DEX": dex} for i, (dex, p) in enumerate(st.session_state.price_history.items())]
            fig = px.line(price_data, x="Time", y="Price", color="DEX", title="Price History")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Balances <i class='fas fa-wallet fa-lg' style='color: #4CAF50;'></i>", unsafe_allow_html=True)
        card(
            title="SOL Balance",
            text=f"{st.session_state.sol_balance:{BALANCE_DECIMALS}} SOL",
            key="sol_balance_card"
        )
        st.markdown(f'<i class="fas fa-solar-coin fa-lg fa-icon"></i>', unsafe_allow_html=True)  
        card(
            title=f"{ticker} Balance",
            text=f"{st.session_state.token_balance:{BALANCE_DECIMALS}}",
            key="token_balance_card"
        )
        st.markdown(f'<i class="fas fa-coins fa-lg fa-icon"></i>', unsafe_allow_html=True)  

    # Trade Controls Card
    st.markdown("### Trade Options <i class='fas fa-exchange-alt fa-lg' style='color: #4CAF50;'></i>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            action = st.radio("Action", ["Buy", "Sell"], key="action", horizontal=True)
        with col2:
            dex = st.selectbox("DEX", list(st.session_state.prices.keys()) if st.session_state.prices else [], key="dex")
        with col3:
            amount = st.number_input("Amount", min_value=0.00000001, value=0.1, format="%.8f", key="amount")
        with col4:
            threshold = st.number_input("Threshold (SOL)", min_value=0.0, value=0.0, format="%.7f", key="threshold")

        trade_button = st.button("Trade", key="trade", help="Execute the trade")
        st.markdown(f'<div class="button-with-icon"><i class="fas fa-handshake fa-lg"></i></div>', unsafe_allow_html=True)
        if trade_button:
            if amount <= 0:
                st.error("Invalid amount.")
            else:
                with st.spinner(f"Executing {action} on {dex}..."):
                    signature = await trade_token(action.lower(), dex, amount, st.session_state.token_mint, st.session_state.prices, 7, ticker)
                    if signature:
                        st.success(f"Trade successful! [View on Explorer](https://explorer.solana.com/tx/{signature})")
                        await update_prices_and_balances(ticker)
                    else:
                        st.error("Trade failed. Check logs for details.")

    # Log Window
    st.markdown("### Logs <i class='fas fa-book fa-lg' style='color: #4CAF50;'></i>", unsafe_allow_html=True)
    log_container = st.empty()
    log_text = ""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            log_text += handler.stream.getvalue() + "\n"
    log_container.write(log_text)

    # Arbitrage Detection
    if st.session_state.prices and len(st.session_state.prices) > 1:
        min_price = min(st.session_state.prices.values())
        max_price = max(st.session_state.prices.values())
        if max_price > min_price * 1.01:  # 1% arbitrage threshold
            st.warning(f"Arbitrage Opportunity: Buy at {min(st.session_state.prices, key=st.session_state.prices.get)} ({min_price:{PRICE_DECIMALS}}), "
                       f"Sell at {max(st.session_state.prices, key=st.session_state.prices.get)} ({max_price:{PRICE_DECIMALS}})")

async def get_jupiter_quote(session, input_mint, output_mint, amount):
    url = (
        f"{JUP_API}/quote?"
        f"inputMint={input_mint}&outputMint={output_mint}&amount={amount}"
        f"&slippageBps={SLIPPAGE_BPS}"
    )
    try:
        async with session.get(url) as response:
            if response.status == 404:
                st.warning(f"Jupiter: No quote available for {input_mint} -> {output_mint}")
                return None
            if response.status != 200:
                st.error(f"Jupiter: Error fetching quote: HTTP {response.status}")
                return None
            data = await response.json()
            return data
    except Exception as e:
        st.error(f"Jupiter: Network error: {e}")
        return None

async def get_raydium_price(session, token_mint):
    try:
        async with session.get(RAYDIUM_API) as response:
            if response.status != 200:
                st.error(f"Raydium: Error fetching price: HTTP {response.status}")
                return None
            data = await response.json()
            price = data.get(str(token_mint))
            if price:
                return float(price)
            st.warning(f"Raydium: No price found for {token_mint}")
            return None
    except Exception as e:
        st.error(f"Raydium: Network error fetching price: {e}")
        return None

async def get_dexscreener_price(session, token_mint):
    url = f"{DEXSCREENER_API}{token_mint}"
    try:
        async with session.get(url) as response:
            if response.status != 200:
                st.error(f"DEX Screener: Error fetching price: HTTP {response.status}")
                return None
            data = await response.json()
            if not data.get("pairs"):
                st.warning(f"DEX Screener: No pairs found for {token_mint}")
                return None
            sol_pairs = [pair for pair in data["pairs"] if pair["chainId"] == "solana" and pair["quoteToken"]["address"] == str(SOL_MINT)]
            prices = {}
            for pair in sol_pairs:
                dex = pair.get("dexId", "Unknown")
                if dex in ["raydium", "orca"]:
                    price_sol = float(pair.get("priceNative", 0))
                    if price_sol > 0 and 0.001 < price_sol < 10:
                        prices[dex] = price_sol
            return prices if prices else None
    except Exception as e:
        st.error(f"DEX Screener: Network error fetching price: {e}")
        return None

async def get_token_decimals(mint):
    decimals = 9
    if str(mint) != str(SOL_MINT):
        try:
            token = AsyncToken(agent.connection, mint, TOKEN_PROGRAM_ID, agent.wallet)
            mint_info = await token.get_mint_info()
            if mint_info is not None:
                decimals = mint_info.decimals
            else:
                st.warning(f"No mint info for {mint}, assuming 9 decimals")
        except Exception as e:
            st.error(f"Failed to fetch decimals for {mint}: {e}, assuming 9 decimals")
    return decimals

async def fetch_token_prices(ticker):
    ticker = ticker.upper()
    if ticker == "SOL":
        st.info("SOL is the native currency. Price is 1 SOL per SOL.")
        return {"N/A": 1.0}, SOL_MINT, 9
    elif ticker == "USDC":
        token_mint = USDC_MINT
    elif ticker == "BTC":
        token_mint = WBTC_MINT
    elif ticker == "JUP":
        token_mint = JUP_MINT
    else:
        try:
            token_data = TokenDataManager.get_token_data_by_ticker(ticker)
            if token_data and token_data.address:
                token_mint = Pubkey.from_string(token_data.address)
                st.info(f"Resolved mint for {ticker}: {token_mint}")
            else:
                st.error(f"No Solana token found for '{ticker}'. Try SOL, USDC, BTC, or JUP.")
                return None, None, None
        except Exception as e:
            st.error(f"Error resolving '{ticker}': {e}")
            return None, None, None

    decimals = await get_token_decimals(token_mint)
    st.info(f"Decimals fetched: {decimals}")

    prices = {}
    async with aiohttp.ClientSession() as session:
        amount = PROBE_AMOUNT if ticker not in ["JUP"] else PROBE_AMOUNT_JUP
        jupiter_quote = await get_jupiter_quote(session, SOL_MINT, token_mint, amount)
        if jupiter_quote and "outAmount" in jupiter_quote:
            token_amount = int(jupiter_quote["outAmount"]) / 10**decimals
            prices["Jupiter"] = (PROBE_AMOUNT if ticker not in ["JUP"] else PROBE_AMOUNT_JUP) / 10**9 / token_amount

        raydium_price = await get_raydium_price(session, token_mint)
        if raydium_price:
            prices["Raydium"] = raydium_price

        dexscreener_prices = await get_dexscreener_price(session, token_mint)
        if dexscreener_prices:
            prices.update(dexscreener_prices)

    # Update price history for charting
    st.session_state.price_history[ticker] = {dex: price for dex, price in prices.items()}

    return prices, token_mint, decimals

async def check_balance(token_mint=None):
    sol_balance = await BalanceFetcher.get_balance(agent)
    token_balance = 0
    if token_mint:
        if str(token_mint) == str(SOL_MINT):
            token_balance = sol_balance
        else:
            try:
                token_balance = await BalanceFetcher.get_balance(agent, token_mint)
            except Exception as e:
                st.error(f"Error fetching token balance: {e}")
                token_balance = 0
    return sol_balance, token_balance

async def trade_token(action, dex, amount, token_mint, prices, decimals, custom_ticker):
    sol_balance, token_balance = await check_balance(token_mint)

    if action == "buy":
        sol_amount = amount * prices[dex]
        if sol_balance < sol_amount:
            st.error("Insufficient SOL balance.")
            return None
        st.info(f"Buying {amount:.{decimals}f} {custom_ticker} for {sol_amount:{PRICE_DECIMALS}} SOL on {dex}")
        signature = await TradeManager.trade(
            agent=agent,
            output_mint=token_mint,
            input_amount=sol_amount,
            input_mint=SOL_MINT,
            slippage_bps=SLIPPAGE_BPS
        )
    elif action == "sell":
        if token_balance < amount:
            st.error("Insufficient token balance.")
            return None
        expected_sol = prices[dex] * amount
        st.info(f"Selling {amount:.{decimals}f} {custom_ticker} for ~{expected_sol:{PRICE_DECIMALS}} SOL on {dex}")
        signature = await TradeManager.trade(
            agent=agent,
            output_mint=SOL_MINT,
            input_amount=amount,
            input_mint=token_mint,
            slippage_bps=SLIPPAGE_BPS
        )
    else:
        st.error("Invalid action.")
        return None

    return signature

async def update_prices_and_balances(ticker):
    prices, token_mint, decimals = await fetch_token_prices(ticker)
    sol_balance, token_balance = await check_balance(token_mint)
    st.session_state.prices = prices
    st.session_state.token_mint = token_mint
    st.session_state.sol_balance = sol_balance
    st.session_state.token_balance = token_balance
    st.rerun()

if __name__ == "__main__":
    if not os.getenv("SOLANA_PRIVATE_KEY") and not st.secrets.get("PRIVATE_KEY"):
        st.error("Please set the SOLANA_PRIVATE_KEY environment variable or configure PRIVATE_KEY in secrets.toml to continue.")
        st.stop()
    asyncio.run(main())