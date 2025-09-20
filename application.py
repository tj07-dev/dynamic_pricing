# import ast
# import os

# import joblib
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import streamlit as st
# from PIL import Image
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Page config
# st.set_page_config(
#     page_title="Dynamic Pricing PlayGround",
#     page_icon="💰",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Custom CSS
# st.markdown(
#     """
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 1.5rem;
#         border-radius: 15px;
#         color: white;
#         text-align: center;
#         margin: 0.5rem 0;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.1);
#     }
#     .product-info {
#         background: #f8f9fa;
#         padding: 1rem;
#         border-radius: 10px;
#         border: 2px solid #e9ecef;
#     }
#     .image-container {
#         border: 3px solid #1f77b4;
#         border-radius: 15px;
#         padding: 10px;
#         background: white;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.1);
#     }
#     .parameter-explanation {
#         background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 0.5rem 0;
#         border-left: 4px solid #1f77b4;
#         font-size: 0.9rem;
#     }
#     .impact-indicator {
#         font-size: 1.2rem;
#         margin: 0.2rem 0;
#         font-weight: bold;
#     }
#     .positive-impact { color: #27ae60; }
#     .negative-impact { color: #e74c3c; }
#     .neutral-impact { color: #f39c12; }
# </style>
# """,
#     unsafe_allow_html=True,
# )


# # Load functions with caching
# @st.cache_data(ttl=3600)
# def load_dataset():
#     try:
#         df = pd.read_csv("enriched_dynamic_pricing_dataset.csv")
#         df["productDisplayName"] = df["productDisplayName"].fillna("Generic Product")

#         def safe_parse_prices(x):
#             try:
#                 if isinstance(x, str):
#                     prices = ast.literal_eval(x)
#                     if isinstance(prices, list) and all(
#                         isinstance(v, (int, float)) for v in prices
#                     ):
#                         return np.mean(prices)
#                 return 500.0
#             except:
#                 return 500.0

#         df["competitor_price_mean"] = df["competitor_prices"].apply(safe_parse_prices)
#         df["competitor_price_mean"] = df["competitor_price_mean"].fillna(
#             df["competitor_price_mean"].median()
#         )

#         for col in ["holiday", "weekday", "weekend"]:
#             df[col] = df[col].fillna(0).astype(int).clip(0, 1)

#         return df
#     except Exception as e:
#         st.error(f"Error loading dataset: {e}")
#         return None


# @st.cache_resource
# def load_model_files():
#     try:
#         model = joblib.load("dynamic_pricing_model.pkl")
#         feature_cols = joblib.load("feature_cols.pkl")
#         return model, feature_cols
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None, None


# @st.cache_data
# def calculate_model_metrics(_model, df_encoded, feature_cols):
#     try:
#         X = df_encoded[feature_cols].fillna(0)
#         y_true = df_encoded["optimal_price"].fillna(500)
#         y_pred = _model.predict(X)

#         mae = mean_absolute_error(y_true, y_pred)
#         mse = mean_squared_error(y_true, y_pred)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(y_true, y_pred)

#         return {
#             "MAE": mae,
#             "MSE": mse,
#             "RMSE": rmse,
#             "R²": r2,
#             "Model Type": type(_model).__name__,
#             "Features Count": len(feature_cols),
#             "Training Samples": len(X),
#         }
#     except Exception as e:
#         st.error(f"Error calculating metrics: {e}")
#         return {}


# @st.cache_data
# def prepare_encoded_data(df_orig, features):
#     df_encoded = pd.get_dummies(
#         df_orig, columns=["articleType", "baseColour", "season", "material", "location"]
#     )
#     missing_cols = set(features) - set(df_encoded.columns)
#     for col in missing_cols:
#         df_encoded[col] = 0
#     df_encoded = df_encoded.reindex(
#         columns=features + ["id", "productDisplayName", "optimal_price"], fill_value=0
#     )
#     return df_encoded


# @st.cache_data
# def get_feature_importance(_model, features):
#     return pd.DataFrame(
#         {"feature": features, "importance": _model.feature_importances_}
#     ).sort_values("importance", ascending=False)


# def load_product_image(image_id):
#     image_path = f"images/{image_id}.jpg"
#     try:
#         if os.path.exists(image_path):
#             img = Image.open(image_path)
#             if img.mode != "RGB":
#                 img = img.convert("RGB")
#             return img
#         else:
#             for ext in [".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
#                 alt_path = f"images/{image_id}{ext}"
#                 if os.path.exists(alt_path):
#                     img = Image.open(alt_path)
#                     if img.mode != "RGB":
#                         img = img.convert("RGB")
#                     return img
#         return None
#     except Exception as e:
#         st.warning(f"Error loading image: {e}")
#         return None


# def calculate_feature_impact(
#     feature_value, feature_importance, base_value=0.5, impact_scale=100
# ):
#     """Calculate the impact of a feature on price based on its importance and deviation from base."""
#     deviation = feature_value - base_value
#     impact = deviation * feature_importance * impact_scale
#     return impact, "positive" if impact > 0 else "negative" if impact < 0 else "neutral"


# def get_parameter_explanation(
#     parameter_name, parameter_value, impact_amount, impact_type
# ):
#     """Generate explanation for parameter impact on price."""
#     explanations = {
#         "demand": f"High demand ({parameter_value:.2f}) signals strong customer interest, creating scarcity perception.",
#         "inventory": f"Low inventory ({parameter_value:.2f}) triggers urgency, allowing premium pricing to maximize profit.",
#         "clicks": f"High engagement ({parameter_value:.2f}) indicates popularity, justifying higher price positioning.",
#         "search_volume": f"Trending searches ({parameter_value:.2f}) show rising demand, supporting price increases.",
#         "reviews_count": f"More reviews ({parameter_value:.2f}) build trust, enabling premium pricing strategies.",
#         "competitor_price_mean": f"Competitor pricing at ₹{parameter_value:.0f} sets market benchmark for positioning.",
#         "ratings": f"Quality rating of {parameter_value:.1f}/5 supports {'premium' if parameter_value > 4 else 'standard'} pricing.",
#         "discount_percentage": f"{parameter_value * 100:.0f}% discount reduces price to drive volume, impacting margins.",
#         "freight_price": f"Shipping cost of ₹{parameter_value:.0f} may be partially passed to customers.",
#         "lag_price": f"Previous price of ₹{parameter_value:.0f} informs trend-based price adjustments.",
#         "product_name_length": f"Detailed naming ({parameter_value * 100:.0f} chars) indicates premium positioning.",
#         "product_description_length": f"Comprehensive description ({parameter_value * 2000:.0f} chars) enhances perceived value.",
#         "product_photos_qty": f"{parameter_value * 10:.0f} photos improve engagement and conversion rates.",
#         "product_weight_g": f"Product weight {parameter_value * 5000:.0f}g affects shipping and perceived quality.",
#         "volume": f"Product volume {parameter_value * 10000:.0f} units influences storage and shipping costs.",
#         "holiday": f"Holiday period ({'Yes' if parameter_value == 1 else 'No'}) boosts demand and pricing power.",
#         "weekday": f"{'Weekday' if parameter_value == 1 else 'Weekend'} timing affects buying behavior patterns.",
#         "weekend": f"{'Weekend' if parameter_value == 1 else 'Weekday'} timing often drives higher conversion rates.",
#     }
#     return explanations.get(
#         parameter_name, f"Feature {parameter_name} with value {parameter_value:.2f}."
#     )


# # Load data and model
# df_original = load_dataset()
# model, feature_cols = load_model_files()

# if df_original is None or model is None:
#     st.error(
#         "❌ Failed to load required files. Please ensure model and data files are available."
#     )
#     st.stop()

# df = prepare_encoded_data(df_original, feature_cols)
# importances = get_feature_importance(model, feature_cols)
# model_metrics = calculate_model_metrics(model, df, feature_cols)

# # Sidebar
# with st.sidebar:
#     st.markdown("### 🎯 Product Selection")

#     search_term = st.text_input(
#         "🔍 Search Products",
#         placeholder="Type product name...",
#         key="sidebar_search_input",
#     )

#     if search_term:
#         filtered_products = df_original[
#             df_original["productDisplayName"].str.contains(
#                 search_term, case=False, na=False
#             )
#         ]
#         if len(filtered_products) > 0:
#             product_options = filtered_products["productDisplayName"].unique()[:50]
#         else:
#             st.warning("No products found")
#             product_options = df_original["productDisplayName"].unique()[:50]
#     else:
#         product_options = df_original["productDisplayName"].unique()[:50]

#     selected_product_name = st.selectbox(
#         "Choose Product", product_options, key="sidebar_product_selectbox"
#     )

#     selected_row = df_original[
#         df_original["productDisplayName"] == selected_product_name
#     ].iloc[0]
#     selected_index = df_original[
#         df_original["productDisplayName"] == selected_product_name
#     ].index[0]

#     st.markdown("### 📊 Quick Actions")
#     use_defaults = st.button(
#         "🔄 Reset to Defaults", type="secondary", key="sidebar_reset_button"
#     )

#     st.markdown("### ⚙️ Display Options")
#     show_explanations = st.toggle(
#         "💡 Show Detailed Explanations", value=True, key="sidebar_explanations_toggle"
#     )
#     show_advanced = st.toggle(
#         "🔧 Advanced Mode (Weights)", value=False, key="sidebar_advanced_toggle"
#     )

# # Main header
# st.markdown(
#     '<div class="main-header">💰 Dynamic Pricing PlayGround</div>',
#     unsafe_allow_html=True,
# )

# # Product info section
# col1, col2, col3 = st.columns([3, 2, 2])

# with col1:
#     st.markdown('<div class="product-info">', unsafe_allow_html=True)
#     st.markdown("### 📦 Product Information")
#     st.write(f"**🏷️ Name:** {selected_row['productDisplayName']}")
#     st.write(f"**📂 Category:** {selected_row['articleType']}")
#     st.write(f"**🎨 Color:** {selected_row['baseColour']}")
#     st.write(f"**🌤️ Season:** {selected_row['season']}")
#     st.write(f"**🧵 Material:** {selected_row['material']}")
#     st.write(f"**📍 Location:** {selected_row['location']}")
#     st.write(f"**🆔 Product ID:** {selected_row['id']}")
#     st.markdown("</div>", unsafe_allow_html=True)

# with col2:
#     st.markdown('<div class="image-container">', unsafe_allow_html=True)
#     product_image = load_product_image(selected_row["id"])
#     if product_image:
#         st.image(
#             product_image, width=200, caption=f"📸 {selected_row['productDisplayName']}"
#         )
#     else:
#         st.markdown(
#             """
#         <div style="text-align: center; padding: 50px; background: #f8f9fa; border-radius: 10px;">
#             <h1 style="font-size: 4rem; color: #ccc;">📷</h1>
#             <p style="color: #666;">Image not available</p>
#             <p style="font-size: 0.8rem; color: #999;">ID: {}</p>
#         </div>
#         """.format(selected_row["id"]),
#             unsafe_allow_html=True,
#         )
#     st.markdown("</div>", unsafe_allow_html=True)

# with col3:
#     current_price = selected_row.get("optimal_price", 500)
#     st.markdown(
#         f"""
#     <div class="metric-card">
#         <h3>💰 Current Price</h3>
#         <h1>${current_price:.0f}</h1>
#     </div>
#     """,
#         unsafe_allow_html=True,
#     )

#     predicted_price_placeholder = st.empty()

# st.divider()

# # Main content
# tab1, tab2, tab3 = st.tabs(
#     [
#         "🎯 Price Optimizer",
#         "📊 Model Analytics",
#         "📈 Dataset Profiling",
#     ]
# )

# with tab1:
#     st.markdown("### 🎛️ Dynamic Parameters")

#     if show_explanations:
#         st.markdown(
#             """
#         <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #1f77b4;">
#             <h4>💡 How Dynamic Pricing Works</h4>
#             <p><strong>Market Dynamics:</strong> The AI analyzes real-time signals like demand and inventory to determine optimal pricing.</p>
#             <p><strong>Competitive Positioning:</strong> Compares with competitors to ensure you're neither too high nor too low.</p>
#             <p><strong>Customer Behavior:</strong> Considers engagement (clicks, searches) and trust signals (ratings, reviews) to justify premium pricing.</p>
#             <p><strong>Operational Factors:</strong> Includes shipping costs and timing (holidays, weekends) that affect pricing strategy.</p>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     # Parameter sections with detailed explanations
#     st.markdown("#### 📈 Market Dynamics (Demand & Supply)")

#     col1, col2 = st.columns(2)

#     with col1:
#         demand = st.slider(
#             "🔥 Demand Level (0-1 normalized)",
#             0.0,
#             1.0,
#             float(selected_row["demand"]),
#             0.01,
#             help="Normalized sales volume; higher signals scarcity.",
#             key="main_demand_slider",
#         )

#         # Demand explanation
#         demand_impact, demand_type = calculate_feature_impact(
#             demand,
#             importances[importances["feature"] == "demand"]["importance"].iloc[0],
#         )

#         st.markdown(
#             f"""
#         <div class="parameter-explanation">
#             <div class="impact-indicator {"positive-impact" if demand_type == "positive" else "negative-impact" if demand_type == "negative" else "neutral-impact"}">
#                 {"📈" if demand_type == "positive" else "📉" if demand_type == "negative" else "➡️"} {demand_type.upper()} IMPACT: ₹{abs(demand_impact):.0f}
#             </div>
#             {get_parameter_explanation("demand", demand, abs(demand_impact), demand_type)}
#             <small><em>💡 Why this matters: High demand allows 10-20% price premium. Low demand may require competitive pricing.</em></small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col2:
#         inventory = st.slider(
#             "📦 Inventory Level (0-1 normalized)",
#             0.0,
#             1.0,
#             float(selected_row["inventory"]),
#             0.01,
#             help="Normalized stock; lower creates scarcity.",
#             key="main_inventory_slider",
#         )

#         # Inventory explanation
#         inventory_impact, inventory_type = calculate_feature_impact(
#             inventory,
#             importances[importances["feature"] == "inventory"]["importance"].iloc[0],
#         )

#         st.markdown(
#             f"""
#         <div class="parameter-explanation">
#             <div class="impact-indicator {"positive-impact" if inventory_type == "positive" else "negative-impact" if inventory_type == "negative" else "neutral-impact"}">
#                 {"📈" if inventory_type == "positive" else "📉" if inventory_type == "negative" else "➡️"} {inventory_type.upper()} IMPACT: ₹{abs(inventory_impact):.0f}
#             </div>
#             {get_parameter_explanation("inventory", inventory, abs(inventory_impact), inventory_type)}
#             <small><em>💡 Why this matters: Low inventory can justify 15-30% price increase due to scarcity principle.</em></small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     st.markdown("---")

#     # Engagement section
#     st.markdown("#### 👥 Customer Engagement (Behavior Signals)")

#     col1, col2 = st.columns(2)

#     with col1:
#         clicks = st.slider(
#             "👆 Click Rate (0-1 normalized)",
#             0.0,
#             1.0,
#             float(selected_row["clicks"]),
#             0.01,
#             help="Normalized page views; high engagement boosts price.",
#             key="main_clicks_slider",
#         )

#         clicks_impact, clicks_type = calculate_feature_impact(
#             clicks,
#             importances[importances["feature"] == "clicks"]["importance"].iloc[0],
#         )

#         st.markdown(
#             f"""
#         <div class="parameter-explanation">
#             <div class="impact-indicator {"positive-impact" if clicks_type == "positive" else "negative-impact" if clicks_type == "negative" else "neutral-impact"}">
#                 {"📈" if clicks_type == "positive" else "📉" if clicks_type == "negative" else "➡️"} {clicks_type.upper()} IMPACT: ₹{abs(clicks_impact):.0f}
#             </div>
#             {get_parameter_explanation("clicks", clicks, abs(clicks_impact), clicks_type)}
#             <small><em>💡 Why this matters: High engagement (clicks > 0.7) indicates strong interest, supporting 5-10% price premium.</em></small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col2:
#         search_volume = st.slider(
#             "🔍 Search Volume (0-1 normalized)",
#             0.0,
#             1.0,
#             float(selected_row["search_volume"]),
#             0.01,
#             help="Normalized search queries; trending raises price.",
#             key="main_search_slider",
#         )

#         search_impact, search_type = calculate_feature_impact(
#             search_volume,
#             importances[importances["feature"] == "search_volume"]["importance"].iloc[
#                 0
#             ],
#         )

#         st.markdown(
#             f"""
#         <div class="parameter-explanation">
#             <div class="impact-indicator {"positive-impact" if search_type == "positive" else "negative-impact" if search_type == "negative" else "neutral-impact"}">
#                 {"📈" if search_type == "positive" else "📉" if search_type == "negative" else "➡️"} {search_type.upper()} IMPACT: ₹{abs(search_impact):.0f}
#             </div>
#             {get_parameter_explanation("search_volume", search_volume, abs(search_impact), search_type)}
#             <small><em>💡 Why this matters: Trending searches (volume > 0.6) signal rising demand, justifying 5-15% price increases.</em></small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     st.markdown("---")

#     # Pricing factors section
#     st.markdown("#### 💰 Pricing & Competitive Factors")

#     col1, col2 = st.columns(2)

#     with col1:
#         comp_price = st.slider(
#             "🏪 Competitor Price Mean",
#             200.0,
#             1000.0,
#             float(df.iloc[selected_index]["competitor_price_mean"]),
#             10.0,
#             help="Average competitor price; sets market benchmark.",
#             key="main_comp_price_slider",
#         )

#         # Competitor price explanation
#         comp_price_impact = (
#             (comp_price - 500)
#             * importances[importances["feature"] == "competitor_price_mean"][
#                 "importance"
#             ].iloc[0]
#             * 0.8
#         )
#         comp_type = "positive" if comp_price_impact > 0 else "negative"

#         st.markdown(
#             f"""
#         <div class="parameter-explanation">
#             <div class="impact-indicator {"positive-impact" if comp_type == "positive" else "negative-impact" if comp_type == "negative" else "neutral-impact"}">
#                 {"📈" if comp_type == "positive" else "📉" if comp_type == "negative" else "➡️"} {comp_type.upper()} IMPACT: ₹{abs(comp_price_impact):.0f}
#             </div>
#             {get_parameter_explanation("competitor_price_mean", comp_price, abs(comp_price_impact), comp_type)}
#             <small><em>💡 Why this matters: Price 5-10% above/below competitors based on your brand positioning and quality perception.</em></small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col2:
#         discount = st.slider(
#             "🏷️ Price Reduction (0-50%)",
#             0.0,
#             0.5,
#             float(selected_row["discount_percentage"]),
#             0.01,
#             help="Extra Price reduction.",
#             key="main_discount_slider",
#         )

#         # Discount explanation
#         discount_impact, discount_type = calculate_feature_impact(
#             discount,
#             importances[importances["feature"] == "discount_percentage"][
#                 "importance"
#             ].iloc[0],
#         )

#         st.markdown(
#             f"""
#         <div class="parameter-explanation">
#             <div class="impact-indicator {"positive-impact" if discount_type == "positive" else "negative-impact" if discount_type == "negative" else "neutral-impact"}">
#                 {"📈" if discount_type == "positive" else "📉" if discount_type == "negative" else "➡️"} {discount_type.upper()} IMPACT: ₹{abs(discount_impact):.0f}
#             </div>
#             {get_parameter_explanation("discount_percentage", discount, abs(discount_impact), discount_type)}
#             <small><em>💡 Why this matters: 10-20% discounts drive volume but reduce margins. Use strategically for inventory clearance.</em></small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     st.markdown("---")

#     # Advanced parameters section (collapsible)
#     with st.expander("🔧 Advanced Parameters", expanded=show_advanced):
#         col1, col2 = st.columns(2)

#         with col1:
#             ratings = st.slider(
#                 "⭐ Customer Ratings (1-5)",
#                 1.0,
#                 5.0,
#                 float(selected_row["ratings"]),
#                 0.1,
#                 help="Average customer rating; higher justifies premium pricing.",
#                 key="main_ratings_slider",
#             )

#             freight_price = st.slider(
#                 "🚚 Freight/Shipping Cost",
#                 0.0,
#                 100.0,
#                 float(selected_row["freight_price"]),
#                 1.0,
#                 help="Shipping costs may be partially passed to customers.",
#                 key="main_freight_slider",
#             )

#             segment = st.selectbox(
#                 "👥 Customer Segment",
#                 ["Retail (Individual)", "B2B (Bulk Corporate)"],
#                 index=0 if selected_row["customer_segment"] == 0 else 1,
#                 help="B2B customers often have higher willingness to pay.",
#                 key="main_segment_selectbox",
#             )

#         with col2:
#             lag_price = st.slider(
#                 "📉 Previous Price Reference",
#                 100.0,
#                 1500.0,
#                 float(selected_row["lag_price"]),
#                 10.0,
#                 help="Last period's price; informs trend adjustments.",
#                 key="main_lag_price_slider",
#             )

#             reviews_count = st.slider(
#                 "💬 Reviews Count (0-1 normalized)",
#                 0.0,
#                 1.0,
#                 float(selected_row["reviews_count"]),
#                 0.01,
#                 help="Normalized review volume; builds trust for premium pricing.",
#                 key="main_reviews_slider",
#             )

#             holiday = st.selectbox(
#                 "🎉 Holiday Period?",
#                 ["No (Regular Day)", "Yes (Holiday Season)"],
#                 index=int(selected_row["holiday"]),
#                 help="Holidays boost demand and pricing power.",
#                 key="main_holiday_selectbox",
#             )

#     # Feature weight controls (Advanced mode only)
#     if show_advanced:
#         st.markdown("### 🎛️ Parameter Priority Weights")
#         st.info(
#             "💡 Adjust parameter influence: 1.0 = normal, >1.0 = amplify, <1.0 = reduce impact on final price."
#         )

#         weight_col1, weight_col2 = st.columns(2)

#         with weight_col1:
#             st.markdown("#### 📊 Market Dynamics Weights")
#             weight_demand = st.slider(
#                 "🔥 Demand Weight", 0.0, 3.0, 1.0, 0.1, key="weight_demand_main"
#             )
#             weight_inventory = st.slider(
#                 "📦 Inventory Weight", 0.0, 3.0, 1.0, 0.1, key="weight_inventory_main"
#             )
#             weight_clicks = st.slider(
#                 "👆 Clicks Weight", 0.0, 3.0, 1.0, 0.1, key="weight_clicks_main"
#             )
#             weight_search_volume = st.slider(
#                 "🔍 Search Volume Weight", 0.0, 3.0, 1.0, 0.1, key="weight_search_main"
#             )

#         with weight_col2:
#             st.markdown("#### 💰 Pricing Strategy Weights")
#             weight_comp_price = st.slider(
#                 "🏪 Competitor Price Weight", 0.0, 3.0, 1.0, 0.1, key="weight_comp_main"
#             )
#             weight_discount = st.slider(
#                 "🏷️ Discount Weight", 0.0, 3.0, 1.0, 0.1, key="weight_discount_main"
#             )
#             weight_freight = st.slider(
#                 "🚚 Freight Weight", 0.0, 3.0, 1.0, 0.1, key="weight_freight_main"
#             )
#             weight_reviews = st.slider(
#                 "💬 Reviews Weight", 0.0, 3.0, 1.0, 0.1, key="weight_reviews_main"
#             )

#     else:
#         # Default weights
#         weight_demand = weight_inventory = weight_clicks = weight_search_volume = 1.0
#         weight_comp_price = weight_discount = weight_freight = weight_reviews = 1.0
#         weight_lag_price = weight_product_name = weight_product_desc = 1.0
#         weight_photos = weight_product_weight = weight_volume = 1.0

#     # Set default values for removed parameters
#     holiday = 0
#     weekday = 1
#     weekend = 0
#     segment = "Retail (0)"

#     # Prediction
#     st.markdown("### 🎯 Price Recommendation")

#     try:
#         input_data = pd.DataFrame(
#             np.zeros((1, len(feature_cols))), columns=feature_cols
#         )

#         # Set feature values with weights applied
#         input_data["demand"] = demand * weight_demand
#         input_data["inventory"] = inventory * weight_inventory
#         input_data["clicks"] = clicks * weight_clicks
#         input_data["search_volume"] = search_volume * weight_search_volume
#         input_data["competitor_price_mean"] = comp_price * weight_comp_price
#         input_data["reviews_count"] = reviews_count * weight_reviews
#         input_data["discount_percentage"] = discount * weight_discount
#         input_data["freight_price"] = freight_price * weight_freight
#         input_data["holiday"] = holiday
#         input_data["weekday"] = weekday
#         input_data["weekend"] = weekend
#         input_data["customer_segment"] = 0  # Always retail for simplicity
#         input_data["lag_price"] = lag_price

#         # Set ratings to product default
#         if "ratings" in feature_cols:
#             input_data["ratings"] = float(selected_row["ratings"])

#         # Set categorical features safely
#         for col in feature_cols:
#             if (
#                 col.startswith("articleType_")
#                 and col == f"articleType_{selected_row['articleType']}"
#             ):
#                 input_data[col] = 1
#             elif (
#                 col.startswith("baseColour_")
#                 and col == f"baseColour_{selected_row['baseColour']}"
#             ):
#                 input_data[col] = 1
#             elif (
#                 col.startswith("season_") and col == f"season_{selected_row['season']}"
#             ):
#                 input_data[col] = 1
#             elif (
#                 col.startswith("material_")
#                 and col == f"material_{selected_row['material']}"
#             ):
#                 input_data[col] = 1
#             elif (
#                 col.startswith("location_")
#                 and col == f"location_{selected_row['location']}"
#             ):
#                 input_data[col] = 1

#         # Predict price
#         predicted_price = model.predict(input_data)[0]
#         predicted_price = max(100, predicted_price)

#         # Calculate price change metrics
#         price_change = predicted_price - current_price
#         price_change_percent = (
#             (price_change / current_price) * 100 if current_price > 0 else 0
#         )

#         # Update predicted price display
#         predicted_price_placeholder.markdown(
#             f"""
#         <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
#             <h3>🎯 Predicted Price</h3>
#             <h1>${predicted_price:.0f}</h1>
#             <small>{"📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"} {price_change_percent:+.1f}% from current</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#         # Price recommendation summary
#         st.markdown("### 📊 Price Analysis")
#         col1, col2, col3, col4 = st.columns(4)

#         with col1:
#             st.metric(
#                 "🎯 Recommended Price",
#                 f"${predicted_price:.0f}",
#                 f"${price_change:+.0f}",
#                 delta_color="normal",
#             )
#         with col2:
#             st.metric(
#                 "📈 Price Change %",
#                 f"{price_change_percent:+.1f}%",
#                 help="Percentage change from current price",
#             )
#         with col3:
#             # Estimated revenue impact (simplified)
#             revenue_impact = price_change * (demand * 100) * weight_demand
#             st.metric(
#                 "💵 Revenue Impact",
#                 f"${revenue_impact:+.0f}",
#                 help="Estimated revenue change per 100 units sold",
#             )
#         with col4:
#             # Profit margin estimate
#             cost_estimate = predicted_price * 0.6  # Assuming 40% margin
#             profit_margin = ((predicted_price - cost_estimate) / predicted_price) * 100
#             st.metric(
#                 "💰 Est. Profit Margin",
#                 f"{profit_margin:.1f}%",
#                 help="Estimated profit margin at recommended price",
#             )

#         # Detailed parameter impact analysis
#         if show_explanations:
#             st.markdown("### 🔍 Parameter Impact Analysis")
#             st.write(
#                 "**How each parameter contributes to the final price recommendation:**"
#             )

#             # Key parameter impacts
#             key_params = [
#                 ("demand", demand, weight_demand, "📈 Demand Signal"),
#                 ("inventory", inventory, weight_inventory, "📦 Supply Constraint"),
#                 ("clicks", clicks, weight_clicks, "👆 Engagement Level"),
#                 (
#                     "search_volume",
#                     search_volume,
#                     weight_search_volume,
#                     "🔍 Market Interest",
#                 ),
#                 (
#                     "comp_price",
#                     comp_price,
#                     weight_comp_price,
#                     "🏪 Competitive Positioning",
#                 ),
#                 ("ratings", ratings, 1.0, "⭐ Quality Perception"),
#                 ("discount", discount, weight_discount, "🏷️ Promotional Strategy"),
#             ]

#             for param_name, param_value, param_weight, param_label in key_params:
#                 if param_name == "comp_price":
#                     impact = (
#                         (comp_price - 500)
#                         * importances[
#                             importances["feature"] == "competitor_price_mean"
#                         ]["importance"].iloc[0]
#                         * 0.8
#                         * param_weight
#                     )
#                 else:
#                     impact, impact_type = calculate_feature_impact(
#                         param_value,
#                         importances[importances["feature"] == param_name][
#                             "importance"
#                         ].iloc[0],
#                     )
#                     impact *= param_weight

#                 impact_symbol = "📈" if impact > 0 else "📉" if impact < 0 else "➡️"
#                 impact_color = (
#                     "positive-impact"
#                     if impact > 0
#                     else "negative-impact"
#                     if impact < 0
#                     else "neutral-impact"
#                 )

#                 st.markdown(
#                     f"""
#                 <div class="parameter-explanation">
#                     <div class="impact-indicator {impact_color}">
#                         {impact_symbol} {param_label}: ₹{abs(impact):.0f} {param_name.replace("_", " ").title()}
#                     </div>
#                     <strong>Value:</strong> {param_value:.2f} × <strong>Weight:</strong> {param_weight:.1f} = <strong>Effective:</strong> {param_value * param_weight:.2f}<br>
#                     {get_parameter_explanation(param_name, param_value, abs(impact), "positive" if impact > 0 else "negative")}
#                     <small><em>💡 Business Insight: This parameter contributes {abs(impact) / abs(price_change) * 100:.0f}% to the total price change.</em></small>
#                 </div>
#                 """,
#                     unsafe_allow_html=True,
#                 )

#         # Price recommendation summary
#         st.markdown("### 💡 Recommendation Summary")

#         if price_change > 0:
#             st.success(
#                 f"✅ **Price Increase Recommended**: ${price_change:+.0f} ({price_change_percent:+.1f}%) - Demand signals support premium positioning."
#             )
#         elif price_change < 0:
#             st.warning(
#                 f"⚠️ **Price Reduction Recommended**: ${price_change:+.0f} ({price_change_percent:+.1f}%) - Competitive or inventory factors suggest adjustment."
#             )
#         else:
#             st.info(
#                 "ℹ️ **Price Stable**: No significant change recommended - Current pricing is optimal."
#             )

#         st.markdown(f"""
#         **🎯 Final Recommendation**: Set price to **${predicted_price:.0f}** for optimal revenue balancing.
#         **📊 Confidence**: {"High" if abs(price_change_percent) < 20 else "Medium" if abs(price_change_percent) < 50 else "Low"}
#         **💼 Strategy**: {"Premium Positioning" if demand > 0.7 and ratings > 4.0 else "Competitive Pricing" if comp_price < 400 else "Market Standard"}
#         """)

#     except Exception as e:
#         st.error(f"❌ Error making prediction: {e}")
#         st.write("Please check model files and dataset compatibility.")

# with tab2:
#     st.markdown("### 📊 Model Performance Metrics")

#     if model_metrics:
#         col1, col2, col3, col4 = st.columns(4)

#         with col1:
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>📏 Mean Absolute Error</h4>
#                 <h2>${model_metrics.get("MAE", 0):.1f}</h2>
#                 <small>Average prediction error</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )
#             st.caption(
#                 "💡 Lower is better. Indicates average price prediction accuracy."
#             )

#         with col2:
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>📐 Root Mean Square Error</h4>
#                 <h2>${model_metrics.get("RMSE", 0):.1f}</h2>
#                 <small>Standard deviation of errors</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )
#             st.caption("💡 Penalizes larger errors more than MAE.")

#         with col3:
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>🎯 R² Score</h4>
#                 <h2>{model_metrics.get("R²", 0):.3f}</h2>
#                 <small>Model explanation power</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )
#             st.caption(
#                 "💡 0.7+ indicates good model fit. Shows how well model explains price variation."
#             )

#         with col4:
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>🔢 Model Specs</h4>
#                 <h2>{model_metrics.get("Features Count", 0)}</h2>
#                 <small>Features used</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )
#             st.caption("💡 Number of input features considered by model.")

#         # Model predictions vs actual
#         st.markdown("### 📈 Prediction vs Actual Analysis")

#         try:
#             # Sample data for visualization
#             sample_size = min(500, len(df))
#             sample_df = df.sample(sample_size, random_state=42)
#             X_sample = sample_df[feature_cols].fillna(0)
#             y_true_sample = sample_df["optimal_price"].fillna(500)
#             y_pred_sample = model.predict(X_sample)

#             col1, col2 = st.columns(2)

#             with col1:
#                 # Scatter plot: Predicted vs Actual
#                 fig_scatter = px.scatter(
#                     x=y_true_sample,
#                     y=y_pred_sample,
#                     title="🎯 Predicted vs Actual Prices",
#                     labels={"x": "Actual Price (₹)", "y": "Predicted Price (₹)"},
#                     trendline="ols",
#                     color_discrete_sequence=["#1f77b4"],
#                 )
#                 fig_scatter.add_shape(
#                     type="line",
#                     line=dict(dash="dash", color="red", width=2),
#                     x0=y_true_sample.min(),
#                     y0=y_true_sample.min(),
#                     x1=y_true_sample.max(),
#                     y1=y_true_sample.max(),
#                 )
#                 fig_scatter.update_layout(
#                     height=400,
#                     xaxis_title="Actual Price (₹)",
#                     yaxis_title="Predicted Price (₹)",
#                 )
#                 st.plotly_chart(fig_scatter, use_container_width=True)
#                 st.caption(
#                     "💡 Perfect predictions would lie on the red dashed line (45°)."
#                 )

#             with col2:
#                 # Residuals plot
#                 residuals = y_true_sample - y_pred_sample
#                 fig_residuals = px.scatter(
#                     x=y_pred_sample,
#                     y=residuals,
#                     title="📉 Residuals Analysis",
#                     labels={
#                         "x": "Predicted Price (₹)",
#                         "y": "Residuals (Actual - Predicted)",
#                     },
#                     color_discrete_sequence=["#ff7f0e"],
#                 )
#                 fig_residuals.add_hline(
#                     y=0, line_dash="dash", line_color="red", width=2
#                 )
#                 fig_residuals.update_layout(height=400)
#                 st.plotly_chart(fig_residuals, use_container_width=True)
#                 st.caption(
#                     "💡 Residuals should be randomly scattered around zero for good model fit."
#                 )

#         except Exception as e:
#             st.error(f"Error creating prediction analysis: {e}")

#     # Model quality summary
#     st.markdown("### 📋 Model Quality Assessment")
#     if model_metrics:
#         quality_assessment = []

#         if model_metrics.get("R²", 0) > 0.7:
#             quality_assessment.append("✅ **Excellent Model Fit** - R² > 0.7")
#         elif model_metrics.get("R²", 0) > 0.5:
#             quality_assessment.append("✅ **Good Model Fit** - R² > 0.5")
#         else:
#             quality_assessment.append(
#                 "⚠️ **Fair Model Fit** - Consider feature engineering"
#             )

#         if model_metrics.get("MAE", 0) < 100:
#             quality_assessment.append("✅ **High Accuracy** - MAE < ₹100")
#         else:
#             quality_assessment.append("⚠️ **Moderate Accuracy** - MAE < ₹150")

#         st.success(" ".join(quality_assessment))

# with tab3:
#     st.markdown("### 📈 Dataset Profiling & Insights")

#     # Basic dataset statistics
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.markdown(
#             f"""
#         <div class="metric-card">
#             <h4>📊 Total Products</h4>
#             <h2>{len(df_original):,}</h2>
#             <small>Unique product entries</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col2:
#         st.markdown(
#             f"""
#         <div class="metric-card">
#             <h4>💰 Average Price</h4>
#             <h2>${df_original["optimal_price"].mean():.0f}</h2>
#             <small>Mean optimal price</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col3:
#         st.markdown(
#             f"""
#         <div class="metric-card">
#             <h4>🏷️ Categories</h4>
#             <h2>{df_original["articleType"].nunique()}</h2>
#             <small>Product categories</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     # Dataset quality metrics
#     st.markdown("### 📋 Dataset Quality Overview")

#     # Missing data analysis
#     missing_data = df_original.isnull().sum()
#     missing_percent = (missing_data / len(df_original)) * 100
#     missing_df = (
#         pd.DataFrame(
#             {
#                 "Column": missing_data.index,
#                 "Missing Count": missing_data.values,
#                 "Missing %": missing_percent.values,
#             }
#         )
#         .sort_values("Missing %", ascending=False)
#         .head(10)
#     )

#     if len(missing_df) > 0 and missing_df["Missing %"].max() > 0:
#         st.markdown("#### ❓ Missing Data (Top 10)")
#         fig_missing = px.bar(
#             missing_df,
#             x="Missing %",
#             y="Column",
#             orientation="h",
#             title="Top 10 Columns with Missing Data",
#             color="Missing %",
#             color_continuous_scale="Reds",
#         )
#         fig_missing.update_layout(height=400)
#         st.plotly_chart(fig_missing, use_container_width=True)
#         st.caption(
#             "💡 Missing data can affect model accuracy. Consider imputation strategies."
#         )
#     else:
#         st.success("✅ **No significant missing data detected!**")

#     # Price distribution by category
#     st.markdown("### 💰 Price Analysis by Category")

#     price_by_category = (
#         df_original.groupby("articleType")["optimal_price"]
#         .agg(["mean", "count"])
#         .sort_values("mean", ascending=False)
#     )
#     price_by_category["category"] = price_by_category.index

#     fig_price_category = px.bar(
#         price_by_category.head(10),
#         x="mean",
#         y="category",
#         orientation="h",
#         title="Average Price by Category (Top 10)",
#         color="mean",
#         color_continuous_scale="Viridis",
#         labels={"mean": "Average Price (₹)", "category": "Product Category"},
#     )
#     fig_price_category.update_layout(height=400)
#     st.plotly_chart(fig_price_category, use_container_width=True)

#     st.caption(
#         "💡 Categories with higher average prices often indicate premium positioning or higher production costs."
#     )

#     # Demand vs Price relationship
#     st.markdown("### 📊 Demand vs Price Relationship")

#     demand_price_corr = df_original[["demand", "optimal_price"]].corr().iloc[0, 1]

#     fig_demand_price = px.scatter(
#         df_original,
#         x="demand",
#         y="optimal_price",
#         title=f"Demand vs Price (Correlation: {demand_price_corr:.3f})",
#         trendline="ols",
#         color_discrete_sequence=["#1f77b4"],
#         labels={"demand": "Demand Level (0-1)", "optimal_price": "Optimal Price (₹)"},
#     )
#     fig_demand_price.update_layout(height=400)
#     st.plotly_chart(fig_demand_price, use_container_width=True)

#     st.markdown(f"""
#     **Correlation Analysis**: Demand and Price correlation is {demand_price_corr:.3f}.
#     {"📈 Positive correlation" if demand_price_corr > 0 else "📉 Negative correlation" if demand_price_corr < 0 else "➡️ No correlation"}.
#     This suggests that {"higher demand supports higher prices" if demand_price_corr > 0 else "demand and price move independently"}.
#     """)

#     # Data quality insights
#     st.markdown("### 🔍 Business Insights from Dataset")

#     insights = []

#     # Price range analysis
#     price_range = (
#         df_original["optimal_price"].max() - df_original["optimal_price"].min()
#     )
#     insights.append(
#         f"💰 **Price Range**: ₹{price_range:.0f} (₹{df_original['optimal_price'].min():.0f} - ₹{df_original['optimal_price'].max():.0f})"
#     )

#     # High demand products
#     high_demand_count = (df_original["demand"] > 0.8).sum()
#     high_demand_pct = (high_demand_count / len(df_original)) * 100
#     insights.append(
#         f"🔥 **High Demand Products**: {high_demand_count:,} ({high_demand_pct:.1f}%) have demand > 0.8 - **Opportunity**: These can support 15-25% premium pricing."
#     )

#     # Low inventory products
#     low_inventory_count = (df_original["inventory"] < 0.2).sum()
#     low_inventory_pct = (low_inventory_count / len(df_original)) * 100
#     insights.append(
#         f"📦 **Low Inventory Products**: {low_inventory_count:,} ({low_inventory_pct:.1f}%) have stock < 0.2 - **Action**: Consider 20-40% price surge for urgency."
#     )

#     # High-rated products
#     high_rated_count = (df_original["ratings"] >= 4.0).sum()
#     high_rated_pct = (high_rated_count / len(df_original)) * 100
#     insights.append(
#         f"⭐ **High Quality Products**: {high_rated_count:,} ({high_rated_pct:.1f}%) have ratings ≥4.0 - **Strategy**: Premium positioning with 10-20% markup."
#     )

#     # Category diversity
#     category_count = df_original["articleType"].nunique()
#     top_category = df_original["articleType"].value_counts().index[0]
#     top_category_pct = (df_original["articleType"] == top_category).mean() * 100
#     insights.append(
#         f"🏷️ **Category Diversity**: {category_count} categories, {top_category} dominates ({top_category_pct:.1f}%) - **Focus**: Optimize pricing for dominant categories."
#     )

#     # Seasonal patterns
#     if "season" in df_original.columns:
#         seasonal_avg = df_original.groupby("season")["optimal_price"].mean()
#         highest_season = seasonal_avg.idxmax()
#         insights.append(
#             f"🌤️ **Seasonal Trends**: {highest_season} has highest average price - **Strategy**: Seasonal premium pricing during peak seasons."
#         )

#     # Display insights
#     for insight in insights:
#         st.info(insight)

#     # Dataset completeness
#     st.markdown("### ✅ Dataset Quality Metrics")

#     completeness_cols = st.columns(4)

#     total_cols = len(df_original.columns)
#     numeric_cols = len(df_original.select_dtypes(include=[np.number]).columns)
#     categorical_cols = len(df_original.select_dtypes(include=["object"]).columns)
#     missing_rate = (
#         df_original.isnull().sum().sum() / (len(df_original) * total_cols)
#     ) * 100

#     with completeness_cols[0]:
#         st.metric("📊 Total Columns", total_cols)
#     with completeness_cols[1]:
#         st.metric("🔢 Numeric Features", numeric_cols)
#     with completeness_cols[2]:
#         st.metric("🏷️ Categorical Features", categorical_cols)
#     with completeness_cols[3]:
#         st.metric("❓ Missing Data Rate", f"{missing_rate:.2f}%")

#     if missing_rate < 5:
#         st.success("✅ **Excellent Data Quality** - Missing data < 5%")
#     elif missing_rate < 15:
#         st.info("ℹ️ **Good Data Quality** - Missing data < 15%, consider imputation")
#     else:
#         st.warning(
#             "⚠️ **Data Quality Concern** - High missing data, review data collection"
#         )

# # Add feature weights section (Advanced mode only)
# if show_advanced:
#     st.markdown("### 🎛️ Parameter Priority Weights")
#     st.info(
#         "💡 Adjust parameter influence: 1.0 = normal, >1.0 = amplify, <1.0 = reduce impact on final price."
#     )

#     weight_col1, weight_col2 = st.columns(2)

#     with weight_col1:
#         st.markdown("#### 📊 Market Dynamics Weights")
#         weight_demand = st.slider(
#             "🔥 Demand Weight", 0.0, 3.0, 1.0, 0.1, key="weight_demand_main"
#         )
#         weight_inventory = st.slider(
#             "📦 Inventory Weight", 0.0, 3.0, 1.0, 0.1, key="weight_inventory_main"
#         )
#         weight_clicks = st.slider(
#             "👆 Clicks Weight", 0.0, 3.0, 1.0, 0.1, key="weight_clicks_main"
#         )
#         weight_search_volume = st.slider(
#             "🔍 Search Volume Weight", 0.0, 3.0, 1.0, 0.1, key="weight_search_main"
#         )

#     with weight_col2:
#         st.markdown("#### 💰 Pricing Strategy Weights")
#         weight_comp_price = st.slider(
#             "🏪 Competitor Price Weight", 0.0, 3.0, 1.0, 0.1, key="weight_comp_main"
#         )
#         weight_discount = st.slider(
#             "🏷️ Discount Weight", 0.0, 3.0, 1.0, 0.1, key="weight_discount_main"
#         )
#         weight_freight = st.slider(
#             "🚚 Freight Weight", 0.0, 3.0, 1.0, 0.1, key="weight_freight_main"
#         )
#         weight_reviews = st.slider(
#             "💬 Reviews Weight", 0.0, 3.0, 1.0, 0.1, key="weight_reviews_main"
#         )

# # End of tab1 content

# with tab2:
#     st.markdown("### 📊 Model Performance & Validation")

#     # Model metrics cards
#     if model_metrics:
#         col1, col2, col3, col4 = st.columns(4)

#         with col1:
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>📏 Mean Absolute Error</h4>
#                 <h2>${model_metrics.get("MAE", 0):.1f}</h2>
#                 <small>Average prediction error</small>
#                 <small>💡 Lower = Better Accuracy</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )

#         with col2:
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>📐 Root Mean Square Error</h4>
#                 <h2>${model_metrics.get("RMSE", 0):.1f}</h2>
#                 <small>Standard deviation of errors</small>
#                 <small>💡 Penalizes large errors</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )

#         with col3:
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>🎯 R² Score</h4>
#                 <h2>{model_metrics.get("R²", 0):.3f}</h2>
#                 <small>Model explanation power</small>
#                 <small>💡 0.7+ = Excellent fit</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )

#         with col4:
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>🔢 Model Specs</h4>
#                 <h2>{model_metrics.get("Features Count", 0)}</h2>
#                 <small>Input features</small>
#                 <small>💡 Random Forest: 50 estimators</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )

#         # Model quality assessment
#         st.markdown("### 📋 Model Quality Assessment")

#         r2_score = model_metrics.get("R²", 0)
#         mae_score = model_metrics.get("MAE", 0)

#         if r2_score > 0.7 and mae_score < 100:
#             st.success(
#                 "✅ **Excellent Model Performance** - High R² and low MAE indicate reliable predictions."
#             )
#         elif r2_score > 0.5:
#             st.info(
#                 "ℹ️ **Good Model Performance** - Solid predictions with room for improvement."
#             )
#         else:
#             st.warning(
#                 "⚠️ **Fair Model Performance** - Predictions acceptable but consider feature engineering."
#             )

#         # Prediction vs Actual visualization
#         st.markdown("### 📈 Prediction Analysis")

#         try:
#             # Sample data for visualization (to avoid RAM issues)
#             sample_size = min(300, len(df))
#             sample_df = df.sample(sample_size, random_state=42)
#             X_sample = sample_df[feature_cols].fillna(0)
#             y_true_sample = sample_df["optimal_price"].fillna(500)
#             y_pred_sample = model.predict(X_sample)

#             col1, col2 = st.columns(2)

#             with col1:
#                 # Scatter plot: Predicted vs Actual
#                 fig_scatter = px.scatter(
#                     x=y_true_sample,
#                     y=y_pred_sample,
#                     title="🎯 Predicted vs Actual Prices",
#                     labels={"x": "Actual Price (₹)", "y": "Predicted Price (₹)"},
#                     trendline="ols",
#                     trendline_color_override="red",
#                     color_discrete_sequence=["#1f77b4"],
#                 )
#                 fig_scatter.add_shape(
#                     type="line",
#                     line=dict(dash="dash", color="red", width=2),
#                     x0=y_true_sample.min(),
#                     y0=y_true_sample.min(),
#                     x1=y_true_sample.max(),
#                     y1=y_true_sample.max(),
#                 )
#                 fig_scatter.update_layout(height=400)
#                 st.plotly_chart(fig_scatter, use_container_width=True)
#                 st.caption(
#                     "💡 Perfect predictions would lie on the red dashed line (45° angle). Points close to this line indicate accurate predictions."
#                 )

#             with col2:
#                 # Residuals plot
#                 residuals = y_true_sample - y_pred_sample
#                 fig_residuals = px.scatter(
#                     x=y_pred_sample,
#                     y=residuals,
#                     title="📉 Residuals Analysis",
#                     labels={
#                         "x": "Predicted Price (₹)",
#                         "y": "Residuals (Actual - Predicted, ₹)",
#                     },
#                     color_discrete_sequence=["#ff7f0e"],
#                 )
#                 fig_residuals.add_hline(
#                     y=0, line_dash="dash", line_color="red", width=2
#                 )
#                 fig_residuals.update_layout(height=400)
#                 st.plotly_chart(fig_residuals, use_container_width=True)
#                 st.caption(
#                     "💡 Residuals should be randomly scattered around zero line. Patterns indicate systematic errors."
#                 )

#             # Prediction error distribution
#             st.markdown("### 📊 Error Distribution")
#             fig_error_hist = px.histogram(
#                 x=residuals,
#                 nbins=30,
#                 title="Distribution of Prediction Errors",
#                 labels={"x": "Prediction Error (₹)"},
#                 color_discrete_sequence=["#e74c3c"],
#             )
#             fig_error_hist.add_vline(x=0, line_dash="dash", line_color="black")
#             fig_error_hist.update_layout(height=400)
#             st.plotly_chart(fig_error_hist, use_container_width=True)
#             st.caption(
#                 "💡 Most errors should cluster around zero. Symmetric distribution indicates unbiased model."
#             )

#         except Exception as e:
#             st.error(f"Error creating prediction analysis: {e}")

#     # Feature importance preview
#     st.markdown("### 🎯 Key Model Drivers")
#     st.write("**Top 5 features influencing price predictions:**")

#     top_features = importances.head(5)
#     for idx, row in top_features.iterrows():
#         importance_pct = row["importance"] * 100
#         st.markdown(
#             f"**{idx + 1}.** {row['feature'].replace('_', ' ').title()}: **{importance_pct:.1f}%** influence"
#         )
#         st.caption(
#             f"💡 This feature explains {importance_pct:.1f}% of price variation in the model."
#         )

# with tab3:
#     st.markdown("### 📈 Dataset Profiling & Business Insights")

#     # Basic dataset statistics
#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         st.markdown(
#             f"""
#         <div class="metric-card">
#             <h4>📊 Total Products</h4>
#             <h2>{len(df_original):,}</h2>
#             <small>Unique product entries</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col2:
#         st.markdown(
#             f"""
#         <div class="metric-card">
#             <h4>💰 Average Price</h4>
#             <h2>${df_original["optimal_price"].mean():.0f}</h2>
#             <small>Mean optimal price</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col3:
#         st.markdown(
#             f"""
#         <div class="metric-card">
#             <h4>🏷️ Categories</h4>
#             <h2>{df_original["articleType"].nunique()}</h2>
#             <small>Product categories</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col4:
#         st.markdown(
#             f"""
#         <div class="metric-card">
#             <h4>⭐ Avg Rating</h4>
#             <h2>{df_original["ratings"].mean():.1f}/5</h2>
#             <small>Customer satisfaction</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     # Missing data analysis
#     st.markdown("### 📋 Data Quality Assessment")

#     missing_data = df_original.isnull().sum()
#     missing_percent = (missing_data / len(df_original)) * 100
#     missing_df = (
#         pd.DataFrame(
#             {
#                 "Column": missing_data.index,
#                 "Missing Count": missing_data.values,
#                 "Missing %": missing_percent.values,
#             }
#         )
#         .sort_values("Missing %", ascending=False)
#         .head(10)
#     )

#     col1, col2 = st.columns(2)

#     with col1:
#         if len(missing_df) > 0 and missing_df["Missing %"].max() > 0:
#             st.markdown("#### ❓ Missing Data Overview")
#             fig_missing = px.bar(
#                 missing_df,
#                 x="Missing %",
#                 y="Column",
#                 orientation="h",
#                 title="Top 10 Columns with Missing Data",
#                 color="Missing %",
#                 color_continuous_scale="Reds",
#                 height=400,
#             )
#             st.plotly_chart(fig_missing, use_container_width=True)
#         else:
#             st.success("✅ **Complete Dataset** - No missing values detected!")

#     with col2:
#         st.markdown("#### 📊 Data Type Distribution")
#         data_types = df_original.dtypes.value_counts()
#         fig_types = px.pie(
#             values=data_types.values,
#             names=data_types.index.astype(str),
#             title="Data Types Distribution",
#             color_discrete_sequence=px.colors.qualitative.Set3,
#         )
#         st.plotly_chart(fig_types, use_container_width=True)

#     # Price analysis by category
#     st.markdown("### 💰 Price Analysis by Category")

#     if "articleType" in df_original.columns:
#         price_by_category = (
#             df_original.groupby("articleType")["optimal_price"]
#             .agg(["mean", "count", "std"])
#             .sort_values("mean", ascending=False)
#         )
#         price_by_category["category"] = price_by_category.index

#         # Top 10 categories by average price
#         top_categories = price_by_category.head(10)

#         fig_price_category = px.bar(
#             top_categories,
#             x="mean",
#             y="category",
#             orientation="h",
#             title="Top 10 Categories by Average Price",
#             color="mean",
#             color_continuous_scale="Viridis",
#             labels={"mean": "Average Price (₹)", "category": "Product Category"},
#             height=400,
#         )
#         st.plotly_chart(fig_price_category, use_container_width=True)

#         st.caption(
#             "💡 Categories with higher average prices often indicate premium positioning or specialized products."
#         )

#         # Category distribution
#         st.markdown("#### 📊 Category Distribution")
#         category_counts = df_original["articleType"].value_counts().head(10)
#         fig_category_dist = px.bar(
#             x=category_counts.values,
#             y=category_counts.index,
#             orientation="h",
#             title="Top 10 Categories by Product Count",
#             color=category_counts.values,
#             color_continuous_scale="Blues",
#             labels={"x": "Product Count", "y": "Category"},
#             height=400,
#         )
#         st.plotly_chart(fig_category_dist, use_container_width=True)

#     # Demand and inventory analysis
#     st.markdown("### 📊 Demand & Inventory Patterns")

#     col1, col2 = st.columns(2)

#     with col1:
#         # Demand distribution
#         fig_demand = px.histogram(
#             df_original,
#             x="demand",
#             nbins=20,
#             title="Demand Level Distribution",
#             labels={"demand": "Demand (0-1 normalized)", "count": "Product Count"},
#             color_discrete_sequence=["#ff7f0e"],
#         )
#         fig_demand.update_layout(height=400)
#         st.plotly_chart(fig_demand, use_container_width=True)

#         st.caption(
#             "💡 Most products have moderate demand (0.2-0.8). High demand (>0.8) products are pricing opportunities."
#         )

#     with col2:
#         # Inventory distribution
#         fig_inventory = px.histogram(
#             df_original,
#             x="inventory",
#             nbins=20,
#             title="Inventory Level Distribution",
#             labels={
#                 "inventory": "Inventory (0-1 normalized)",
#                 "count": "Product Count",
#             },
#             color_discrete_sequence=["#2ca02c"],
#         )
#         fig_inventory.update_layout(height=400)
#         st.plotly_chart(fig_inventory, use_container_width=True)

#         st.caption(
#             "💡 Low inventory (<0.2) creates urgency pricing opportunities. High inventory may require discounts."
#         )

#     # Demand vs Price correlation
#     if "demand" in df_original.columns and "optimal_price" in df_original.columns:
#         demand_price_corr = df_original[["demand", "optimal_price"]].corr().iloc[0, 1]

#         st.markdown("### 🔗 Demand-Price Relationship")
#         st.markdown(f"**Correlation Coefficient**: {demand_price_corr:.3f}")

#         if demand_price_corr > 0.3:
#             st.success(
#                 "📈 **Strong Positive Relationship** - Higher demand supports higher prices"
#             )
#         elif demand_price_corr > 0:
#             st.info("📈 **Moderate Positive Relationship** - Demand influences pricing")
#         else:
#             st.warning(
#                 "➡️ **Weak/No Relationship** - Demand may not directly drive pricing"
#             )

#         fig_demand_price = px.scatter(
#             df_original,
#             x="demand",
#             y="optimal_price",
#             title=f"Demand vs Price (Correlation: {demand_price_corr:.3f})",
#             trendline="ols",
#             trendline_color_override="red",
#             color_discrete_sequence=["#1f77b4"],
#             labels={
#                 "demand": "Demand Level (0-1)",
#                 "optimal_price": "Optimal Price (₹)",
#             },
#             height=400,
#         )
#         st.plotly_chart(fig_demand_price, use_container_width=True)

#     # Business insights section
#     st.markdown("### 🎯 Actionable Business Insights")

#     insights_container = st.container()

#     with insights_container:
#         col1, col2 = st.columns(2)

#         with col1:
#             # High opportunity products
#             high_demand_low_inventory = df_original[
#                 (df_original["demand"] > 0.7) & (df_original["inventory"] < 0.3)
#             ]
#             pricing_opportunities = len(high_demand_low_inventory)

#             st.markdown("#### 🚀 Pricing Opportunities")
#             st.info(
#                 f"**{pricing_opportunities} products** have high demand AND low inventory"
#             )
#             if pricing_opportunities > 0:
#                 st.success(
#                     f"💡 **Recommendation**: Consider **15-25% price increase** for these {pricing_opportunities} products to maximize urgency revenue."
#                 )
#                 st.caption(
#                     f"Examples: {', '.join(high_demand_low_inventory['productDisplayName'].head(3).tolist())}"
#                 )

#         with col2:
#             # Discount candidates
#             high_inventory_low_demand = df_original[
#                 (df_original["inventory"] > 0.7) & (df_original["demand"] < 0.3)
#             ]
#             discount_candidates = len(high_inventory_low_demand)

#             st.markdown("#### 🏷️ Discount Candidates")
#             st.warning(
#                 f"**{discount_candidates} products** have high inventory AND low demand"
#             )
#             if discount_candidates > 0:
#                 st.info(
#                     f"💡 **Recommendation**: Consider **10-20% discount** for these {discount_candidates} products to clear inventory."
#                 )
#                 st.caption(
#                     f"Examples: {', '.join(high_inventory_low_demand['productDisplayName'].head(3).tolist())}"
#                 )

#         # Premium positioning candidates
#         high_quality_low_competition = df_original[
#             (df_original["ratings"] >= 4.0)
#             & (df_original["competitor_price_mean"] < 400)
#         ]
#         premium_candidates = len(high_quality_low_competition)

#         st.markdown("#### ⭐ Premium Positioning")
#         st.success(
#             f"**{premium_candidates} products** have high ratings AND low competitor prices"
#         )
#         if premium_candidates > 0:
#             st.info(
#                 f"💡 **Recommendation**: Position these {premium_candidates} products as premium with **10-15% markup**."
#             )
#             st.caption(
#                 f"Examples: {', '.join(high_quality_low_competition['productDisplayName'].head(3).tolist())}"
#             )

#         # Seasonal opportunities
#         if "season" in df_original.columns:
#             seasonal_analysis = (
#                 df_original.groupby("season")
#                 .agg({"optimal_price": "mean", "demand": "mean", "inventory": "mean"})
#                 .round(2)
#             )

#             st.markdown("#### 🌤️ Seasonal Pricing Strategy")
#             st.dataframe(seasonal_analysis)
#             st.caption(
#                 "💡 Higher demand seasons support premium pricing. Low inventory seasons need urgent action."
#             )

#         # Geographic pricing
#         if "location" in df_original.columns:
#             geo_analysis = (
#                 df_original.groupby("location")
#                 .agg({"optimal_price": "mean", "demand": "mean"})
#                 .round(2)
#             )

#             st.markdown("#### 🌍 Geographic Pricing Strategy")
#             st.dataframe(geo_analysis)
#             st.caption(
#                 "💡 Higher prices in premium markets (US, EU). Competitive pricing in price-sensitive markets (Asia)."
#             )

import ast
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
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

# Custom CSS
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
        background-clip: text;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load functions with caching
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
            df[col] = df[col].fillna(0).astype(int).clip(0, 1)

        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("dynamic_pricing_model.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
        return model, feature_cols
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def calculate_model_metrics(_model, df_encoded, feature_cols):
    try:
        X = df_encoded[feature_cols].fillna(0)
        y_true = df_encoded["optimal_price"].fillna(500)
        y_pred = _model.predict(X)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2,
            "Model Type": type(_model).__name__,
            "Features Count": len(feature_cols),
            "Training Samples": len(X),
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {}


@st.cache_data
def prepare_encoded_data(df_orig, features):
    df_encoded = pd.get_dummies(
        df_orig, columns=["articleType", "baseColour", "season", "material", "location"]
    )
    missing_cols = set(features) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0
    df_encoded = df_encoded.reindex(
        columns=features + ["id", "productDisplayName", "optimal_price"], fill_value=0
    )
    return df_encoded


@st.cache_data
def get_feature_importance(_model, features):
    return pd.DataFrame(
        {"feature": features, "importance": _model.feature_importances_}
    ).sort_values("importance", ascending=False)


def load_product_image(image_id):
    image_path = f"images/{image_id}.jpg"
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        else:
            for ext in [".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                alt_path = f"images/{image_id}{ext}"
                if os.path.exists(alt_path):
                    img = Image.open(alt_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    return img
        return None
    except Exception as e:
        st.warning(f"Error loading image: {e}")
        return None


# Load data and model
df_original = load_dataset()
model, feature_cols = load_model_files()

if df_original is None or model is None:
    st.error(
        "❌ Failed to load required files. Please ensure model and data files are available."
    )
    st.stop()

df = prepare_encoded_data(df_original, feature_cols)
importances = get_feature_importance(model, feature_cols)
model_metrics = calculate_model_metrics(model, df, feature_cols)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Product Selection")

    search_term = st.text_input(
        "🔍 Search Products",
        placeholder="Type product name...",
        key="sidebar_search_input",
    )

    if search_term:
        filtered_products = df_original[
            df_original["productDisplayName"].str.contains(
                search_term, case=False, na=False
            )
        ]
        if len(filtered_products) > 0:
            product_options = filtered_products["productDisplayName"].unique()[:50]
        else:
            st.warning("No products found")
            product_options = df_original["productDisplayName"].unique()[:50]
    else:
        product_options = df_original["productDisplayName"].unique()[:50]

    selected_product_name = st.selectbox(
        "Choose Product", product_options, key="sidebar_product_selectbox"
    )

    selected_row = df_original[
        df_original["productDisplayName"] == selected_product_name
    ].iloc[0]
    selected_index = df_original[
        df_original["productDisplayName"] == selected_product_name
    ].index[0]

    # st.markdown("### 📊 Quick Actions")
    use_defaults = st.button(
        "🔄 Reset to Defaults", type="secondary", key="sidebar_reset_button"
    )

    # st.markdown("### ⚙️ Display Options")
    show_explanations = st.toggle(
        "💡 Show Explanations", value=True, key="sidebar_explanations_toggle"
    )
    show_advanced = st.toggle(
        "🔧 Advanced Mode", value=False, key="sidebar_advanced_toggle"
    )

# Main header
st.markdown(
    '<div class="main-header">💰 Dynamic Pricing PlayGround</div>',
    unsafe_allow_html=True,
)

# Product info section
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.markdown('<div class="product-info">', unsafe_allow_html=True)
    st.markdown("### 📦 Product Details")
    st.write(f"**🏷️ Name:** {selected_row['productDisplayName']}")
    st.write(f"**📂 Category:** {selected_row['articleType']}")
    st.write(f"**🎨 Color:** {selected_row['baseColour']}")
    st.write(f"**🌤️ Season:** {selected_row['season']}")
    st.write(f"**🧵 Material:** {selected_row['material']}")
    st.write(f"**📍 Location:** {selected_row['location']}")
    st.write(f"**🆔 Product ID:** {selected_row['id']}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    product_image = load_product_image(selected_row["id"])
    if product_image:
        st.image(
            product_image, width=200, caption=f"📸 {selected_row['productDisplayName']}"
        )
    else:
        st.markdown(
            """
        <div style="text-align: center; padding: 50px; background: #f8f9fa; border-radius: 10px;">
            <h1 style="font-size: 4rem; color: #ccc;">📷</h1>
            <p style="color: #666;">Image not available</p>
            <p style="font-size: 0.8rem; color: #999;">ID: {}</p>
        </div>
        """.format(selected_row["id"]),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    current_price = selected_row.get("optimal_price", 500)
    st.markdown(
        f"""
    <div class="metric-card">
        <h3>💰 Current Price</h3>
        <h1>${current_price:.0f}</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )

    predicted_price_placeholder = st.empty()

st.divider()

# Main content
tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Price Optimizer",
        "📊 Model Analytics",
        "📈 Dataset Profiling",
    ]
)

with tab1:
    st.markdown("### 🎛️ Dynamic Parameters")

    if show_explanations:
        st.markdown(
            """
        <div style="background:  #000; padding: 1.5rem; border-radius: 10px; color: white; margin: 1rem 0;">
            <h4>💡 How It Works</h4>
            <p>Adjust parameters to see real-time price recommendations. The AI considers market dynamics, regional factors, and competitive positioning.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Market Dynamics")
        demand = st.slider(
            "🔥 Demand Level",
            0.0,
            1.0,
            float(selected_row["demand"]),
            0.01,
            help="Higher demand allows premium pricing",
            key="main_demand_slider",
        )
        inventory = st.slider(
            "📦 Inventory Level",
            0.0,
            1.0,
            float(selected_row["inventory"]),
            0.01,
            help="Lower inventory creates scarcity",
            key="main_inventory_slider",
        )
        clicks = st.slider(
            "👆 Click Rate",
            0.0,
            1.0,
            float(selected_row["clicks"]),
            0.01,
            help="Higher engagement indicates popularity",
            key="main_clicks_slider",
        )
        search_volume = st.slider(
            "🔍 Search Volume",
            0.0,
            1.0,
            float(selected_row["search_volume"]),
            0.01,
            help="Trending products command higher prices",
            key="main_search_slider",
        )

    with col2:
        st.markdown("#### 💰 Pricing Factors")
        comp_price = st.slider(
            "🏪 Competitor Avg Price",
            200.0,
            1000.0,
            float(df.iloc[selected_index]["competitor_price_mean"]),
            10.0,
            help="Competitive positioning reference",
            key="main_comp_price_slider",
        )
        discount = st.slider(
            "🏷️ Price Reduction",
            0.0,
            0.5,
            float(selected_row["discount_percentage"]),
            0.01,
            help="Promotional discount reduces price",
            key="main_discount_slider",
        )
        freight_price = st.slider(
            "🚚 Shipping Cost",
            0.0,
            100.0,
            float(selected_row["freight_price"]),
            1.0,
            help="Shipping costs may be passed to customer",
            key="main_freight_slider",
        )
    # Weight controls
    if show_advanced:
        st.markdown("### 🎛️ Parameter Weights")
        st.info(
            "💡 Adjust parameter influence: 1.0 = normal, >1.0 = amplify, <1.0 = reduce"
        )

        weight_col1, weight_col2 = st.columns(2)

        with weight_col1:
            st.markdown("#### 📊 Market Weights")
            weight_demand = st.slider(
                "🔥 Demand Weight", 0.0, 3.0, 1.0, 0.1, key="weight_demand_main"
            )
            weight_inventory = st.slider(
                "📦 Inventory Weight", 0.0, 3.0, 1.0, 0.1, key="weight_inventory_main"
            )
            weight_clicks = st.slider(
                "👆 Clicks Weight", 0.0, 3.0, 1.0, 0.1, key="weight_clicks_main"
            )
            weight_search_volume = st.slider(
                "🔍 Search Volume Weight", 0.0, 3.0, 1.0, 0.1, key="weight_search_main"
            )

        with weight_col2:
            st.markdown("#### 📈 Engagement Weights")
            weight_reviews = st.slider(
                "💬 Reviews Weight", 0.0, 3.0, 1.0, 0.1, key="weight_reviews_main"
            )

            st.markdown("#### 💰 Pricing Weights")
            weight_comp_price = st.slider(
                "🏪 Competitor Price Weight", 0.0, 3.0, 1.0, 0.1, key="weight_comp_main"
            )
            weight_discount = st.slider(
                "🏷️ Discount Weight", 0.0, 3.0, 1.0, 0.1, key="weight_discount_main"
            )
            weight_freight = st.slider(
                "🚚 Freight Weight", 0.0, 3.0, 1.0, 0.1, key="weight_freight_main"
            )

    else:
        # Default weights
        weight_demand = weight_inventory = weight_clicks = weight_search_volume = 1.0
        weight_comp_price = weight_discount = weight_freight = weight_reviews = 1.0
        weight_lag_price = weight_product_name = weight_product_desc = 1.0
        weight_photos = weight_product_weight = weight_volume = 1.0

    # Set default values for removed parameters
    holiday = int(selected_row["holiday"])
    weekday = int(selected_row["weekday"])
    weekend = int(selected_row["weekend"])
    segment = "Retail (0)"

    # Prediction
    st.markdown("### 🎯 Price Recommendation")

    try:
        input_data = pd.DataFrame(
            np.zeros((1, len(feature_cols))), columns=feature_cols
        )

        # Set feature values with weights applied
        input_data["demand"] = demand * weight_demand
        input_data["inventory"] = inventory * weight_inventory
        input_data["clicks"] = clicks * weight_clicks
        input_data["search_volume"] = search_volume * weight_search_volume
        input_data["competitor_price_mean"] = comp_price * weight_comp_price
        # input_data["reviews_count"] = reviews_count * weight_reviews
        input_data["discount_percentage"] = discount * weight_discount
        input_data["freight_price"] = freight_price * weight_freight
        input_data["holiday"] = holiday
        input_data["weekday"] = weekday
        input_data["weekend"] = weekend
        input_data["customer_segment"] = 0  # Always retail
        # input_data["lag_price"] = lag_price * weight_lag_price

        # Set ratings to product default
        if "ratings" in feature_cols:
            input_data["ratings"] = float(selected_row["ratings"])

        # Set categorical features
        for col in feature_cols:
            if (
                col.startswith("articleType_")
                and col == f"articleType_{selected_row['articleType']}"
            ):
                input_data[col] = 1
            elif (
                col.startswith("baseColour_")
                and col == f"baseColour_{selected_row['baseColour']}"
            ):
                input_data[col] = 1
            elif (
                col.startswith("season_") and col == f"season_{selected_row['season']}"
            ):
                input_data[col] = 1
            elif (
                col.startswith("material_")
                and col == f"material_{selected_row['material']}"
            ):
                input_data[col] = 1
            elif (
                col.startswith("location_")
                and col == f"location_{selected_row['location']}"
            ):
                input_data[col] = 1

        predicted_price = model.predict(input_data)[0]
        predicted_price = max(100, predicted_price)

        # Update predicted price display
        predicted_price_placeholder.markdown(
            f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h3>🎯 Predicted Price</h3>
            <h1>${predicted_price:.0f}</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Calculate changes
        price_change = predicted_price - current_price
        price_change_percent = (price_change / current_price) * 100

        # Display results
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🎯 Recommended Price",
                f"${predicted_price:.0f}",
                f"${price_change:.0f}",
            )
        with col2:
            st.metric("📈 Price Change", f"{price_change_percent:.1f}%")
        with col3:
            revenue_impact = price_change * demand * 100 * weight_demand
            st.metric("💵 Revenue Impact", f"${revenue_impact:.0f}")
        with col4:
            profit_margin = (
                (predicted_price - (predicted_price * 0.7)) / predicted_price
            ) * 100
            st.metric("💰 Profit Margin", f"{profit_margin:.1f}%")

    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")

with tab2:
    st.markdown("### 📊 Model Performance Metrics")

    if model_metrics:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>📏 MAE</h4>
                <h2>${model_metrics.get("MAE", 0):.1f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>📐 RMSE</h4>
                <h2>${model_metrics.get("RMSE", 0):.1f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>🎯 R²</h4>
                <h2>{model_metrics.get("R²", 0):.3f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # with col4:
        #     st.markdown(
        #         f"""
        #     <div class="metric-card">
        #         <h4>🔢 Features</h4>
        #         <h2>{model_metrics.get("Features Count", 0)}</h2>
        #     </div>
        #     """,
        #         unsafe_allow_html=True,
        #     )

    # Model predictions vs actual
    st.markdown("### 📈 Prediction Analysis")

    try:
        # Sample data for visualization
        sample_size = min(500, len(df))
        sample_df = df.sample(sample_size, random_state=42)
        X_sample = sample_df[feature_cols].fillna(0)
        y_true_sample = sample_df["optimal_price"].fillna(500)
        y_pred_sample = model.predict(X_sample)

        col1, col2 = st.columns(2)

        with col1:
            # Scatter plot: Predicted vs Actual
            fig_scatter = px.scatter(
                x=y_true_sample,
                y=y_pred_sample,
                title="Predicted vs Actual Prices",
                labels={"x": "Actual Price ($)", "y": "Predicted Price ($)"},
                trendline="ols",
            )
            fig_scatter.add_shape(
                type="line",
                line=dict(dash="dash", color="red"),
                x0=y_true_sample.min(),
                y0=y_true_sample.min(),
                x1=y_true_sample.max(),
                y1=y_true_sample.max(),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            # Residuals plot
            residuals = y_true_sample - y_pred_sample
            fig_residuals = px.scatter(
                x=y_pred_sample,
                y=residuals,
                title="Residuals Plot",
                labels={"x": "Predicted Price ($)", "y": "Residuals ($)"},
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating prediction analysis: {e}")

with tab3:
    st.markdown("### 📈 Dataset Profiling")

    # Custom dataset profiling function
    def create_dataset_profile(df):
        profile_data = {}
        profile_data["Shape"] = df.shape
        profile_data["Memory Usage"] = (
            f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        return {
            "basic_info": profile_data,
            "missing_data": missing_data[missing_data > 0],
            "missing_percent": missing_percent[missing_percent > 0],
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }

    profile = create_dataset_profile(df_original)

    # Basic statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>📊 Dataset Shape</h4>
            <h2>{profile["basic_info"]["Shape"][0]:,}</h2>
            <small>Rows × {profile["basic_info"]["Shape"][1]} Columns</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>💾 Memory Usage</h4>
            <h2>{profile["basic_info"]["Memory Usage"]}</h2>
            <small>Total dataset size</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        missing_percent = (
            df_original.isnull().sum().sum()
            / (len(df_original) * len(df_original.columns))
        ) * 100
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>❓ Missing Data</h4>
            <h2>{missing_percent:.1f}%</h2>
            <small>Overall completeness</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Column analysis
    st.markdown("### 📋 Column Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔢 Numeric Columns")
        numeric_stats = df_original[profile["numeric_cols"]].describe()
        st.dataframe(numeric_stats)

    with col2:
        st.markdown("#### 🏷️ Categorical Columns")
        cat_info = []
        for col in profile["categorical_cols"]:
            unique_count = df_original[col].nunique()
            most_common = (
                df_original[col].mode().iloc[0]
                if len(df_original[col].mode()) > 0
                else "N/A"
            )
            cat_info.append(
                {
                    "Column": col,
                    "Unique Values": unique_count,
                    "Most Common": most_common,
                }
            )
        st.dataframe(pd.DataFrame(cat_info))

    # Missing data visualization
    if len(profile["missing_data"]) > 0:
        st.markdown("### ❓ Missing Data Analysis")

        missing_df = pd.DataFrame(
            {
                "Column": profile["missing_data"].index,
                "Missing Count": profile["missing_data"].values,
                "Missing Percentage": profile["missing_percent"].values,
            }
        )

        fig_missing = px.bar(
            missing_df.head(20),
            x="Missing Percentage",
            y="Column",
            orientation="h",
            title="Top 20 Columns with Missing Data",
            color="Missing Percentage",
            color_continuous_scale="reds",
        )
        st.plotly_chart(fig_missing, use_container_width=True)

    # Price distribution analysis
    st.markdown("### 💰 Price Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Price histogram
        fig_hist = px.histogram(
            df_original,
            x="optimal_price",
            title="Price Distribution",
            nbins=50,
            color_discrete_sequence=["#1f77b4"],
        )
        fig_hist.update_layout(xaxis_title="Price ($)", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Price by category
        price_by_category = (
            df_original.groupby("articleType")["optimal_price"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig_category = px.bar(
            x=price_by_category.values,
            y=price_by_category.index,
            orientation="h",
            title="Average Price by Category (Top 10)",
            color=price_by_category.values,
            color_continuous_scale="viridis",
        )
        fig_category.update_layout(
            xaxis_title="Average Price ($)", yaxis_title="Category"
        )
        st.plotly_chart(fig_category, use_container_width=True)

    # Correlation analysis
    st.markdown("### 🔗 Feature Correlations")

    numeric_cols_for_corr = [
        "demand",
        "clicks",
        "search_volume",
        "inventory",
        # "reviews_count",
        "ratings",
        "discount_percentage",
        "freight_price",
        "optimal_price",
    ]

    available_cols = [
        col for col in numeric_cols_for_corr if col in df_original.columns
    ]

    if len(available_cols) >= 3:
        corr_matrix = df_original[available_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto",
            text_auto=True,
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Data quality insights
    st.markdown("### 🔍 Data Quality Insights")

    insights = []
    price_range = (
        df_original["optimal_price"].max() - df_original["optimal_price"].min()
    )
    insights.append(
        f"💰 Price range spans ${price_range:.0f} (${df_original['optimal_price'].min():.0f} - ${df_original['optimal_price'].max():.0f})"
    )

    high_demand_products = (df_original["demand"] > 0.8).sum()
    insights.append(
        f"🔥 {high_demand_products} products ({high_demand_products / len(df_original) * 100:.1f}%) have high demand (>0.8)"
    )

    low_inventory_products = (df_original["inventory"] < 0.2).sum()
    insights.append(
        f"📦 {low_inventory_products} products ({low_inventory_products / len(df_original) * 100:.1f}%) have low inventory (<0.2)"
    )

    high_rated_products = (df_original["ratings"] >= 4.0).sum()
    insights.append(
        f"⭐ {high_rated_products} products ({high_rated_products / len(df_original) * 100:.1f}%) have high ratings (≥4.0)"
    )

    category_count = df_original["articleType"].nunique()
    insights.append(f"🏷️ Dataset contains {category_count} different product categories")

    for insight in insights:
        st.info(insight)

with tab3:
    st.markdown("### 📈 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>📊 Total Products</h4>
            <h2>{len(df_original):,}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>💰 Avg Price</h4>
            <h2>${df_original["optimal_price"].mean():.0f}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>🏷️ Categories</h4>
            <h2>{df_original["articleType"].nunique()}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )
