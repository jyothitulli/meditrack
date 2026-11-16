"""
MediTrack Prescriptive Decision Support Engine (50% Milestone)
Translates predictive stockout risk probabilities into actionable clinical recommendations:
1. Recommended Order Quantities (ROQ) based on Lead Time, Safety Stock, and Consumption Velocity.
2. Intra-District Lateral Stock Transfer Rebalancing (pairing deficit facilities with surplus facilities).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    get_risk_level
)


def calculate_reorder_quantity(
    current_stock: float,
    monthly_consumption: float,
    consumption_std: float = 0.0,
    lead_time_days: int = 14,
    review_period_days: int = 30,
    service_level_z: float = 1.65
) -> dict:
    """
    Computes standard healthcare replenishment Order-Up-To Level (OUL) and Recommended Order Quantity (ROQ).
    
    Formula:
        Daily Demand (D) = monthly_consumption / 30.0
        Safety Stock (SS) = Z * (consumption_std / sqrt(30)) * sqrt(lead_time_days)
        Order-Up-To Level (OUL) = (lead_time_days + review_period_days) * D + SS
        ROQ = max(0, OUL - current_stock)
    """
    daily_consumption = max(0.0, float(monthly_consumption)) / 30.0
    daily_std = max(0.0, float(consumption_std)) / np.sqrt(30.0)
    
    # Safety stock for target service level (default 95% -> Z=1.65)
    safety_stock = service_level_z * daily_std * np.sqrt(lead_time_days)
    # Minimum buffer: at least 7 days of average demand
    safety_stock = max(safety_stock, 7.0 * daily_consumption)
    
    order_up_to_level = (lead_time_days + review_period_days) * daily_consumption + safety_stock
    roq = max(0.0, order_up_to_level - float(current_stock))
    
    days_of_stock = float(current_stock) / (daily_consumption + 1e-5)
    
    return {
        "daily_consumption": round(daily_consumption, 2),
        "safety_stock": int(np.ceil(safety_stock)),
        "order_up_to_level": int(np.ceil(order_up_to_level)),
        "recommended_order_qty": int(np.ceil(roq)),
        "estimated_days_of_stock": round(days_of_stock, 1)
    }


def find_district_rebalancing_transfers(
    snapshot_df: pd.DataFrame,
    target_hf_pk: int,
    target_product_id: int,
    min_surplus_days: float = 60.0,
    reserve_days_buffer: float = 30.0
) -> list:
    """
    Identifies candidate donor health facilities within the same district that have surplus stock
    for the specified medicine to perform an emergency intra-district lateral transfer.
    
    Args:
        snapshot_df: DataFrame containing the latest cross-sectional snapshot across all facilities.
        target_hf_pk: Facility ID of the clinic facing stockout risk.
        target_product_id: Medicine product ID.
        min_surplus_days: Minimum days of stock required for a facility to be considered a donor.
        reserve_days_buffer: Buffer days donor must keep for its own patient population.
        
    Returns:
        List of candidate donor transfer recommendations.
    """
    # 1. Locate recipient facility record
    recip = snapshot_df[
        (snapshot_df["hf_pk"] == target_hf_pk) & 
        (snapshot_df["productID"] == target_product_id)
    ]
    if recip.empty:
        return []
    
    recip_row = recip.iloc[0]
    district = recip_row.get("district", None)
    recip_stock = float(recip_row.get("current_stock", 0.0))
    recip_monthly_c = float(recip_row.get("consumption_roll3_mean", recip_row.get("current_consumption", 10.0)))
    recip_daily_c = max(0.1, recip_monthly_c / 30.0)
    
    # Recipient deficit to reach 30 days
    needed_qty = max(0.0, (30.0 * recip_daily_c) - recip_stock)
    if needed_qty <= 0:
        return []
        
    # 2. Filter donor candidates: same district, same medicine, different facility
    candidates = snapshot_df[
        (snapshot_df["district"] == district) &
        (snapshot_df["productID"] == target_product_id) &
        (snapshot_df["hf_pk"] != target_hf_pk) &
        (snapshot_df["estimated_days_of_stock"] >= min_surplus_days)
    ].copy()
    
    if candidates.empty:
        return []
        
    # Exclude donors that have high predicted risk
    if "predicted_risk" in candidates.columns:
        candidates = candidates[candidates["predicted_risk"] < RISK_THRESHOLD_LOW]
        
    transfers = []
    for _, donor in candidates.iterrows():
        donor_stock = float(donor.get("current_stock", 0.0))
        donor_monthly_c = float(donor.get("consumption_roll3_mean", donor.get("current_consumption", 10.0)))
        donor_daily_c = max(0.1, donor_monthly_c / 30.0)
        
        # Safe surplus donor can spare without dropping below reserve buffer
        donor_reserve = reserve_days_buffer * donor_daily_c
        spare_capacity = max(0.0, donor_stock - donor_reserve)
        
        if spare_capacity >= 5.0:  # Meaningful transfer volume
            suggested_transfer = int(min(needed_qty, spare_capacity))
            transfers.append({
                "donor_hf_pk": int(donor["hf_pk"]),
                "donor_facility_name": str(donor.get("name2", f"Facility #{donor['hf_pk']}")),
                "donor_district": str(district),
                "donor_current_stock": int(donor_stock),
                "donor_days_of_stock": round(float(donor["estimated_days_of_stock"]), 1),
                "suggested_transfer_qty": suggested_transfer,
                "donor_post_transfer_days": round((donor_stock - suggested_transfer) / donor_daily_c, 1),
                "recipient_post_transfer_days": round((recip_stock + suggested_transfer) / recip_daily_c, 1)
            })
            
    # Sort candidates by surplus days descending
    transfers.sort(key=lambda x: x["donor_days_of_stock"], reverse=True)
    return transfers
