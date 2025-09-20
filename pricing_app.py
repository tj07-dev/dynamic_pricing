import ast
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config(
    page_title="Dynamic Pricing PlayGround",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .product-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    .image-container {
        border: 3px solid #1f77b4;
        border-radius: 15px;
        padding: 10px;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .signal-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        border-left: 3px solid #1f77b4;
        font-size: 0.9rem;
    }
    .signal-high { border-left-color: #27ae60 !important; }
    .signal-medium { border-left-color: #f39c12 !important; }
    .signal-low { border-left-color: #e74c3c !important; }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #11998e, #38ef7d);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def load_dataset():
    try:
        df = pd.read_csv("enriched_dynamic_pricing_dataset.csv")
        df["productDisplayName"] = df["productDisplayName"].fillna("Generic Product")

        def safe_parse_prices(x):
            try:
                if isinstance(x, str):
                    prices = ast.literal_eval(x)
                    if isinstance(prices, list) and all(
                        isinstance(v, (int, float)) for v in prices
                    ):
                        return np.mean(prices)
                return 500.0
            except:
                return 500.0

        df["competitor_price_mean"] = df["competitor_prices"].apply(safe_parse_prices)
        df["competitor_price_mean"] = df["competitor_price_mean"].fillna(
            df["competitor_price_mean"].median()
        )

        for col in ["holiday", "weekday", "weekend"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Ensure optimal_price is numeric
        df["optimal_price"] = pd.to_numeric(
            df["optimal_price"], errors="coerce"
        ).fillna(500)

        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("dynamic_pricing_model1.pkl")
        artifacts = joblib.load("feature_cols1.pkl")
        return model, artifacts
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def load_product_image(image_id):
    try:
        image_path = f"images/{image_id}.jpg"
        if os.path.exists(image_path):
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        return None
    except:
        return None


def prepare_simple_features(selected_row, artifacts, weights=None):
    """Simplified feature preparation matching trained model exactly."""
    try:
        selected_features = artifacts["selected_features_list"]
        n_features = len(selected_features)

        # Create feature vector
        feature_vector = np.zeros(n_features)
        feature_mapping = {name: idx for idx, name in enumerate(selected_features)}

        # Extract and weight features
        row_dict = selected_row.to_dict()

        if weights is None:
            weights = {feat: 1.0 for feat in selected_features}

        for orig_feature in row_dict:
            if orig_feature in feature_mapping:
                feature_idx = feature_mapping[orig_feature]
                value = float(row_dict[orig_feature])
                weighted_value = value * weights.get(orig_feature, 1.0)
                feature_vector[feature_idx] = weighted_value

        # Handle seasonal encoding
        season = row_dict.get("season", "Summer")
        season_feature = f"season_{season}"
        if season_feature in feature_mapping:
            season_idx = feature_mapping[season_feature]
            feature_vector[season_idx] = weights.get(season_feature, 3.0)

        # Handle timing
        holiday_val = float(row_dict.get("holiday", 0))
        weekend_val = float(row_dict.get("weekend", 0))

        if "holiday" in feature_mapping:
            holiday_idx = feature_mapping["holiday"]
            feature_vector[holiday_idx] = holiday_val * weights.get("holiday", 9.0)

        if "weekend" in feature_mapping:
            weekend_idx = feature_mapping["weekend"]
            feature_vector[weekend_idx] = weekend_val * weights.get("weekend", 6.0)

        # Clip values
        feature_vector = np.clip(feature_vector, -5, 5)

        return feature_vector.reshape(1, -1)

    except Exception as e:
        st.error(f"Feature preparation error: {e}")
        return np.zeros((1, len(artifacts["selected_features_list"])))


def calculate_feature_impact(value, importance, base_value=0.5):
    """Calculate feature impact for explanations."""
    try:
        normalized_value = np.tanh(value / 2) if abs(value) > 2 else value
        deviation = normalized_value - base_value
        impact = deviation * importance * 150
        return (
            impact,
            "positive" if impact > 0 else "negative" if impact < 0 else "neutral",
        )
    except:
        return 0, "neutral"


def get_parameter_explanation(param_name, param_value, impact, impact_type):
    """Generate detailed parameter explanations."""
    explanations = {
        "demand": f"High demand ({param_value:.2f}) creates scarcity perception, justifying premium pricing strategies. Impact: ${impact:+.0f}",
        "inventory": f"Low inventory ({param_value:.2f}) triggers urgency pricing. Lower stock = higher price tolerance. Impact: ${impact:+.0f}",
        "clicks": f"Strong engagement ({param_value:.2f}) indicates product popularity, supporting higher price positioning. Impact: ${impact:+.0f}",
        "search_volume": f"Rising search volume ({param_value:.2f}) signals growing market demand. Trending products command premiums. Impact: ${impact:+.0f}",
        "competitor_price_mean": f"Competitors pricing at ${param_value:.0f} sets market benchmark. Strategic positioning relative to this level. Impact: ${impact:+.0f}",
        "ratings": f"Quality rating {param_value:.1f}/5 establishes premium perception. Higher ratings = greater pricing flexibility. Impact: ${impact:+.0f}",
        "reviews_count": f"Social proof from {param_value:.2f} reviews builds consumer trust, enabling confident premium pricing. Impact: ${impact:+.0f}",
        "holiday": f"Holiday timing ({'Yes' if param_value > 0 else 'No'}) boosts demand elasticity by 20-30%. Impact: ${impact:+.0f}",
        "weekend": f"Weekend effect ({'Yes' if param_value > 0 else 'No'}) drives higher conversion rates and pricing power. Impact: ${impact:+.0f}",
    }
    return explanations.get(
        param_name,
        f"Parameter {param_name} ({param_value:.2f}) contributes ${impact:+.0f} to pricing intelligence.",
    )


def get_signal_status(value, signal_type):
    """Get visual status indicators."""
    try:
        if signal_type == "demand":
            return (
                "🔥 HIGH",
                "success-card"
                if value > 0.7
                else "warning-card"
                if value > 0.4
                else "metric-card",
            )
        elif signal_type == "inventory":
            return (
                "⚡ LOW",
                "success-card"
                if value < 0.3
                else "warning-card"
                if value < 0.7
                else "metric-card",
            )
        elif signal_type == "clicks":
            return (
                "📈 STRONG",
                "success-card"
                if value > 0.6
                else "warning-card"
                if value > 0.3
                else "metric-card",
            )
        elif signal_type == "search_volume":
            return (
                "🔍 TRENDING",
                "success-card"
                if value > 0.6
                else "warning-card"
                if value > 0.3
                else "metric-card",
            )
        elif signal_type == "ratings":
            return (
                f"{value:.1f}/5",
                "success-card"
                if value > 4.0
                else "warning-card"
                if value > 3.5
                else "metric-card",
            )
        elif signal_type == "competitor_price":
            ratio = value / 500 if 500 > 0 else 1.0
            return "⚖️ BALANCED", "success-card" if 0.9 < ratio < 1.1 else "warning-card"
        else:
            return f"{value:.2f}", "metric-card"
    except:
        return "➡️ NORMAL", "metric-card"


# Load data and model
try:
    df_original = load_dataset()
    model, artifacts = load_model_files()

    if df_original is None or model is None:
        st.error("❌ Failed to load model files.")
        st.stop()

    selected_features = artifacts["selected_features_list"]

    # Calculate model metrics for display
    try:
        X_display = df_original[selected_features].fillna(0)
        y_true = df_original["optimal_price"].fillna(500)
        y_pred = model.predict(X_display)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        model_metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "Features Count": len(selected_features),
            "Training Samples": len(df_original),
        }
    except:
        model_metrics = {
            "MAE": 50,
            "RMSE": 65,
            "R²": 0.65,
            "Features Count": len(selected_features),
        }

except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()

# Sidebar - Product Selection with Search
with st.sidebar:
    st.markdown("### 🎯 Product Selection")

    # Search functionality
    search_term = st.text_input("🔍 Search Products", placeholder="Search by name...")

    if search_term:
        filtered_df = df_original[
            df_original["productDisplayName"].str.contains(
                search_term, case=False, na=False
            )
        ]
        if len(filtered_df) > 0:
            product_options = filtered_df["productDisplayName"].unique()[:25]
        else:
            st.warning("No products found. Showing popular items.")
            product_options = (
                df_original["productDisplayName"].value_counts().head(25).index.tolist()
            )
    else:
        # Show most popular products
        product_options = (
            df_original["productDisplayName"].value_counts().head(25).index.tolist()
        )

    selected_product = st.selectbox(
        "Choose Product", product_options, key="product_selector"
    )
    selected_row = df_original[
        df_original["productDisplayName"] == selected_product
    ].iloc[0]

    # Product preview
    with st.expander(f"ℹ️ {selected_row['articleType']} Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Season:** {selected_row['season']}")
            st.write(f"**Material:** {selected_row['material']}")
            st.write(f"**Market:** {selected_row['location']}")
        with col2:
            st.write(f"**Demand:** {selected_row['demand']:.2f}")
            st.write(f"**Stock:** {selected_row['inventory']:.2f}")
            st.write(f"**Rating:** {selected_row['ratings']:.1f}★")

    # Controls
    st.markdown("### ⚙️ Intelligence Mode")
    show_advanced = st.toggle(
        "🔧 Advanced Calibration", value=False, key="advanced_mode"
    )
    show_explanations = st.toggle(
        "💡 Signal Intelligence", value=True, key="show_explain"
    )

    if st.button("🔄 Reset All Signals", type="secondary"):
        st.rerun()

# Enhanced Main Header
st.markdown(
    """
<div class="main-header">
    🚀 Dynamic Pricing PlayGround 
</div>
""",
    unsafe_allow_html=True,
)

# Product Intelligence Display
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.markdown('<div class="product-info">', unsafe_allow_html=True)
    st.markdown("### 🧠 Product Intelligence Profile")

    # Enhanced product information
    st.write(f"**🏷️ {selected_row['productDisplayName']}**")
    st.write(f"*{selected_row['articleType']}* in *{selected_row['baseColour']}*")

    info_row1 = st.columns([1, 1])
    with info_row1[0]:
        st.write(f"**Season:** {selected_row['season']}")
        st.write(f"**Material:** {selected_row['material']}")
    with info_row1[1]:
        st.write(f"**Market:** {selected_row['location']}")
        st.write(f"**ID:** #{int(selected_row['id'])}")

    # Intelligence preview
    intel_preview = st.columns(3)
    with intel_preview[0]:
        demand_status, demand_class = get_signal_status(
            selected_row["demand"], "demand"
        )
        st.markdown(
            f"""
        <div class="signal-card {demand_class.replace("card", "")}" style="padding:0.5rem;margin:0.25rem 0">
            <div style="font-size:1.1rem">🔥 {demand_status}</div>
            <div style="font-weight:bold;color:#1f77b4">{selected_row["demand"]:.2f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with intel_preview[1]:
        inv_status, inv_class = get_signal_status(
            selected_row["inventory"], "inventory"
        )
        st.markdown(
            f"""
        <div class="signal-card {inv_class.replace("card", "")}" style="padding:0.5rem;margin:0.25rem 0">
            <div style="font-size:1.1rem">📦 {inv_status}</div>
            <div style="font-weight:bold;color:#1f77b4">{selected_row["inventory"]:.2f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with intel_preview[2]:
        rating_status, rating_class = get_signal_status(
            selected_row["ratings"], "ratings"
        )
        st.markdown(
            f"""
        <div class="signal-card {rating_class.replace("card", "")}" style="padding:0.5rem;margin:0.25rem 0">
            <div style="font-size:1.1rem">{rating_status}</div>
            <div style="font-weight:bold;color:#1f77b4">{selected_row["ratings"]:.1f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    img = load_product_image(selected_row["id"])
    if img:
        st.image(img, width=180, use_column_width=None, caption="")
    else:
        st.markdown(
            """
        <div style="text-align:center;padding:20px;height:100%">
            <h2 style="color:#ddd;margin:0">📦</h2>
            <p style="color:#999;margin:0.5rem 0 0 0;font-size:0.9rem">{}</p>
        </div>
        """.format(selected_row["articleType"]),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    current_price = max(50, float(selected_row.get("optimal_price", 500)))
    price_class = "success-card" if current_price > 300 else "metric-card"

    st.markdown(
        f"""
    <div class="{price_class}" style="height:50%;padding:1rem;display:flex;flex-direction:column;justify-content:center">
        <h3 style="margin:0 0 0.5rem 0;font-size:1rem">💰 Current Price</h3>
        <h1 style="margin:0;font-size:2.8rem;line-height:1">${current_price:.0f}</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Prediction placeholder
    predicted_price_placeholder = st.empty()

st.divider()

# Elite Price Optimizer
st.markdown("### Price Optimizer")

# Intelligence Briefing
if show_explanations:
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
                padding: 1.2rem; border-radius: 12px; margin: 1rem 0; 
                border-left: 5px solid #1f77b4;">
        <h4 style="margin-top: 0;">💡 How Dynamic Pricing Works</h4>
        <p><strong>Market Dynamics:</strong> The AI analyzes real-time signals like demand and inventory to determine optimal pricing.</p>
<p><strong>Competitive Positioning:</strong> Compares with competitors to ensure you're neither too high nor too low.</p>
<p><strong>Customer Behavior:</strong> Considers engagement (clicks, searches) and trust signals (ratings, reviews) to justify premium pricing.</p>
<p><strong>Operational Factors:</strong> Includes shipping costs and timing (holidays, weekends) that affect pricing strategy.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Core Signal Controls
st.markdown("#### 🔥 Core Market Intelligence")

signal_col1, signal_col2 = st.columns(2)

with signal_col1:
    st.markdown("**Demand Intelligence**")
    demand = st.slider(
        "Demand Power",
        0.0,
        1.0,
        float(selected_row["demand"]),
        0.01,
        help="15x weight • High demand creates 20-30% premium opportunity",
        key="demand_intel",
    )

    # Demand signal status with explanation
    demand_status, demand_class = get_signal_status(demand, "demand")
    demand_impact, demand_type = calculate_feature_impact(
        demand, 0.15 if "demand" in selected_features else 0.05
    )

    demand_class_css = demand_class.replace("card", "")
    st.markdown(
        f"""
    <div class="signal-card {demand_class_css}" style="margin-top:0.5rem">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:1.1rem">{demand_status}</span>
            <span style="font-weight:bold;color:#1f77b4">{demand:.2f}</span>
        </div>
        <div style="font-size:0.85rem;margin-top:0.25rem;color:#666">
            {get_parameter_explanation("demand", demand, abs(demand_impact), demand_type)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("**Inventory Intelligence**")
    inventory = st.slider(
        "Stock Scarcity",
        0.0,
        1.0,
        float(selected_row["inventory"]),
        0.01,
        help="12x weight • Low inventory = scarcity pricing power",
        key="inventory_intel",
    )

    inv_status, inv_class = get_signal_status(inventory, "inventory")
    inv_impact, inv_type = calculate_feature_impact(
        1 - inventory,
        0.12 if "inventory" in selected_features else 0.05,  # Inverse for scarcity
    )

    inv_class_css = inv_class.replace("card", "")
    st.markdown(
        f"""
    <div class="signal-card {inv_class_css}" style="margin-top:0.5rem">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:1.1rem">{inv_status}</span>
            <span style="font-weight:bold;color:#1f77b4">{inventory:.2f}</span>
        </div>
        <div style="font-size:0.85rem;margin-top:0.25rem;color:#666">
            {get_parameter_explanation("inventory", inventory, abs(inv_impact), inv_type)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with signal_col2:
    st.markdown("**Engagement Intelligence**")
    clicks = st.slider(
        "Click Velocity",
        0.0,
        1.0,
        float(selected_row["clicks"]),
        0.01,
        help="8x weight • Strong engagement validates premium positioning",
        key="clicks_intel",
    )

    click_status, click_class = get_signal_status(clicks, "clicks")
    click_impact, click_type = calculate_feature_impact(
        clicks, 0.08 if "clicks" in selected_features else 0.05
    )

    click_class_css = click_class.replace("card", "")
    st.markdown(
        f"""
    <div class="signal-card {click_class_css}" style="margin-top:0.5rem">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:1.1rem">{click_status}</span>
            <span style="font-weight:bold;color:#1f77b4">{clicks:.2f}</span>
        </div>
        <div style="font-size:0.85rem;margin-top:0.25rem;color:#666">
            {get_parameter_explanation("clicks", clicks, abs(click_impact), click_type)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("**Search Intelligence**")
    search_volume = st.slider(
        "Search Momentum",
        0.0,
        1.0,
        float(selected_row["search_volume"]),
        0.01,
        help="8x weight • Rising searches signal growing demand",
        key="search_intel",
    )

    search_status, search_class = get_signal_status(search_volume, "search_volume")
    search_impact, search_type = calculate_feature_impact(
        search_volume, 0.08 if "search_volume" in selected_features else 0.05
    )

    search_class_css = search_class.replace("card", "")
    st.markdown(
        f"""
    <div class="signal-card {search_class_css}" style="margin-top:0.5rem">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:1.1rem">{search_status}</span>
            <span style="font-weight:bold;color:#1f77b4">{search_volume:.2f}</span>
        </div>
        <div style="font-size:0.85rem;margin-top:0.25rem;color:#666">
            {get_parameter_explanation("search_volume", search_volume, abs(search_impact), search_type)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Competitive Intelligence Row
st.markdown("#### 🏪 Competitive Intelligence")
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.markdown("**Market Benchmark**")
    comp_price = st.slider(
        "Competitor Price",
        100.0,
        1500.0,
        float(selected_row["competitor_price_mean"]),
        25.0,
        help="10x weight • Strategic positioning against market leaders",
        key="comp_price_intel",
    )

    # Competitive positioning analysis
    price_position = comp_price / current_price if current_price > 0 else 1.0
    position_status, position_class = get_signal_status(
        price_position, "competitor_price"
    )
    comp_impact, comp_type = calculate_feature_impact(
        price_position, 0.10 if "competitor_price_mean" in selected_features else 0.05
    )

    position_class_css = position_class.replace("card", "")
    st.markdown(
        f"""
    <div class="signal-card {position_class_css}" style="margin-top:0.5rem">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:1.1rem">🏪 {position_status}</span>
            <span style="font-weight:bold;color:#1f77b4">${comp_price:.0f}</span>
        </div>
        <div style="font-size:0.85rem;margin-top:0.25rem;color:#666">
            {get_parameter_explanation("competitor_price_mean", comp_price, abs(comp_impact), comp_type)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with comp_col2:
    st.markdown("**Quality Intelligence**")
    ratings = st.slider(
        "Customer Rating",
        1.0,
        5.0,
        float(selected_row["ratings"]),
        0.1,
        help="4x weight • Higher ratings justify premium positioning",
        key="ratings_intel",
    )

    rating_status, rating_class = get_signal_status(ratings, "ratings")
    rating_impact, rating_type = calculate_feature_impact(
        ratings / 5, 0.04 if "ratings" in selected_features else 0.05
    )

    rating_class_css = rating_class.replace("card", "")
    st.markdown(
        f"""
    <div class="signal-card {rating_class_css}" style="margin-top:0.5rem">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:1.1rem">⭐ {rating_status}</span>
            <span style="font-weight:bold;color:#1f77b4">{ratings:.1f}</span>
        </div>
        <div style="font-size:0.85rem;margin-top:0.25rem;color:#666">
            {get_parameter_explanation("ratings", ratings, abs(rating_impact), rating_type)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Timing Intelligence
st.markdown("#### ⏰ Timing Intelligence")
timing_col = st.columns(1)

with timing_col[0]:
    timing_options = {
        "Normal Day": {"holiday": 0.0, "weekend": 0.0, "label": "Standard Timing"},
        "Holiday/Event": {"holiday": 1.0, "weekend": 0.0, "label": "Peak Season Boost"},
        "Weekend Effect": {
            "holiday": 0.0,
            "weekend": 1.0,
            "label": "Weekend Conversion",
        },
    }

    selected_timing = st.selectbox(
        "Timing Context",
        list(timing_options.keys()),
        index=0,
        help="Peak timing unlocks 20-30% additional pricing authority",
        key="timing_intel",
    )

    timing_data = timing_options[selected_timing]
    holiday_val = timing_data["holiday"]
    weekend_val = timing_data["weekend"]
    timing_label = timing_data["label"]

    timing_class = (
        "success-card" if holiday_val > 0 or weekend_val > 0 else "metric-card"
    )
    timing_icon = "🎉" if holiday_val > 0 else "🗓️" if weekend_val > 0 else "📅"

    timing_impact, timing_type = calculate_feature_impact(
        holiday_val + weekend_val, 0.09 if "holiday" in selected_features else 0.05
    )

    st.markdown(
        f"""
    <div class="{timing_class.replace("card", "")}" style="padding:0.75rem;margin-top:0.5rem">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:1.1rem">{timing_icon} {timing_label}</span>
            <span style="font-weight:bold;color:#1f77b4">{holiday_val + weekend_val:.1f}x</span>
        </div>
        <div style="font-size:0.85rem;margin-top:0.25rem;color:#666">
            {get_parameter_explanation("holiday", holiday_val + weekend_val, abs(timing_impact), timing_type)}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Advanced Calibration
if show_advanced:
    st.markdown("### ⚖️ Intelligence Calibration")
    st.info(
        "🔧 Fine-tune signal priorities. Higher multipliers amplify market intelligence impact."
    )

    calib_col1, calib_col2 = st.columns(2)

    with calib_col1:
        demand_calib = st.slider(
            "Demand Multiplier",
            5.0,
            25.0,
            15.0,
            1.0,
            help="Primary intelligence driver - scarcity creates value",
            key="calib_demand",
        )
        inventory_calib = st.slider(
            "Scarcity Multiplier",
            5.0,
            20.0,
            12.0,
            1.0,
            help="Low stock = pricing power - inverse relationship",
            key="calib_inventory",
        )

    with calib_col2:
        comp_calib = st.slider(
            "Competition Multiplier",
            5.0,
            15.0,
            10.0,
            1.0,
            help="Market positioning intelligence",
            key="calib_comp",
        )
        engagement_calib = st.slider(
            "Engagement Multiplier",
            3.0,
            12.0,
            8.0,
            1.0,
            help="Customer interest validation",
            key="calib_engagement",
        )

    custom_weights = {
        "demand": demand_calib,
        "inventory": inventory_calib,
        "clicks": engagement_calib,
        "search_volume": engagement_calib,
        "competitor_price_mean": comp_calib,
        "holiday": 9.0,
        "ratings": 4.0,
        "reviews_count": 3.5,
        "weekend": 6.0,
        "month": 2.5,
        "weekday": 2.0,
        "customer_segment": 0.8,
    }
else:
    custom_weights = None

# Elite Intelligence Engine
# st.markdown("### 🎯 Elite Intelligence Recommendation")

# Intelligence Processing
with st.spinner("🧠 Processing elite intelligence signals..."):
    try:
        # Prepare enhanced row with current values
        enhanced_row = selected_row.copy()
        enhanced_row["demand"] = demand
        enhanced_row["inventory"] = inventory
        enhanced_row["clicks"] = clicks
        enhanced_row["search_volume"] = search_volume
        enhanced_row["competitor_price_mean"] = comp_price
        enhanced_row["ratings"] = ratings
        enhanced_row["holiday"] = holiday_val
        enhanced_row["weekend"] = weekend_val
        enhanced_row["reviews_count"] = float(selected_row.get("reviews_count", 0.5))

        # Generate elite prediction
        input_features = prepare_simple_features(
            enhanced_row, artifacts, custom_weights
        )
        elite_price = model.predict(input_features)[0]
        elite_price = max(50, min(2000, elite_price))

        # Intelligence metrics
        scarcity_intel = demand / max(inventory, 0.01)
        engagement_intel = clicks + search_volume
        timing_intel = holiday_val * 2.5 + weekend_val * 1.8
        quality_intel = ratings / 5.0

        intel_score = (
            scarcity_intel * 0.3
            + engagement_intel * 0.25
            + timing_intel * 0.2
            + quality_intel * 0.25
        )
        intel_level = (
            "ELITE"
            if intel_score > 2.5
            else "PREMIUM"
            if intel_score > 1.5
            else "OPTIMAL"
        )
        intel_color = (
            "#11998e"
            if intel_level == "ELITE"
            else "#27ae60"
            if intel_level == "PREMIUM"
            else "#1f77b4"
        )

        # Price change analysis
        price_change = elite_price - current_price
        change_pct = (price_change / current_price * 100) if current_price > 0 else 0
        change_direction = (
            "INCREASE"
            if price_change > 0
            else "DECREASE"
            if price_change < 0
            else "ADJUSTMENT"
        )
        change_icon = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
        change_class = (
            "success-card"
            if price_change > 0
            else "warning-card"
            if price_change < 0
            else "metric-card"
        )

        # Elite Recommendation Display
        predicted_price_placeholder.markdown(
            f"""
        <div class="{change_class}" style="padding:1.5rem;text-align:center;position:relative">
            <h3 style="margin:0 0 0.5rem 0;font-size:1.1rem">{change_icon} Elite Intelligence</h3>
            <h1 style="font-size:3.2rem">${elite_price:.0f}</h1>
            <div style="font-size:1.2rem;font-weight:bold;margin-bottom:0.75rem">
                Predicted Price
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;font-size:0.9rem">
                <div style="text-align:center;padding:0.25rem">
                    <div style="font-weight:bold;color:{"#" + intel_color}">{scarcity_intel:.1f}x</div>
                    <div style="opacity:0.9">Scarcity</div>
                </div>
                <div style="text-align:center;padding:0.25rem">
                    <div style="font-weight:bold">{engagement_intel:.1f}</div>
                    <div style="opacity:0.9">Engagement</div>
                </div>
                <div style="text-align:center;padding:0.25rem">
                    <div style="font-weight:bold">{timing_intel:.1f}</div>
                    <div style="opacity:0.9">Timing</div>
                </div>
                <div style="text-align:center;padding:0.25rem">
                    <div style="font-weight:bold">{quality_intel:.1f}</div>
                    <div style="opacity:0.9">Quality</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Intelligence Impact Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            delta_color = (
                "normal"
                if abs(change_pct) < 3
                else "inverse"
                if change_pct < 0
                else "off"
            )
            st.metric(
                "🎯 Recommended Price",
                f"${elite_price:.0f}",
                f"${price_change:+.0f}",
                delta_color=delta_color,
            )

        with metric_col2:
            st.metric("📈 Price Change", f"{change_pct:+.1f}%")

        with metric_col3:
            volume_estimate = max(1000, min(5000, demand * 6000))
            revenue_impact = price_change * volume_estimate
            revenue_color = "normal" if revenue_impact > 0 else "inverse"
            st.metric(
                "💰 Revenue Intelligence",
                f"${revenue_impact:+,.0f}",
                delta_color=revenue_color,
            )

        # with metric_col4:
        #     quality_factor = min(1.0, ratings / 5 * 1.2)
        #     cogs_rate = 0.4 - (quality_factor * 0.15)
        #     gross_margin = max(20, min(70, ((1 - cogs_rate) * 100)))
        #     st.metric("📊 Elite Margin", f"{gross_margin:.0f}%")

    except Exception as e:
        st.error(f"❌ Intelligence Engine Error: {str(e)}")
        st.exception(e)
