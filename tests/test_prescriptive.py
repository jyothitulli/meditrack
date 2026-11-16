"""
Unit tests for Prescriptive Analytics (50% Milestone):
- Recommended Reorder Quantity calculation
- Intra-District Lateral Stock Transfer Rebalancing
"""
import pytest
import pandas as pd
import numpy as np
from src.ml.prescriptive import calculate_reorder_quantity, find_district_rebalancing_transfers


def test_calculate_reorder_quantity_deficit():
    # Deficit scenario: 300 units/mo (~10/day), 14d lead time, only 15 units remaining (1.5 days)
    res = calculate_reorder_quantity(
        current_stock=15.0,
        monthly_consumption=300.0,
        consumption_std=20.0,
        lead_time_days=14,
        review_period_days=30
    )
    
    assert res["daily_consumption"] == 10.0
    assert res["estimated_days_of_stock"] == 1.5
    assert res["safety_stock"] >= 70  # At least 7 days of demand buffer
    assert res["recommended_order_qty"] > 400  # Must cover lead time + review period + safety stock


def test_calculate_reorder_quantity_surplus():
    # Surplus scenario: 30 units/mo (~1/day), current stock is 5,000 units
    res = calculate_reorder_quantity(
        current_stock=5000.0,
        monthly_consumption=30.0,
        lead_time_days=14,
        review_period_days=30
    )
    assert res["recommended_order_qty"] == 0
    assert res["estimated_days_of_stock"] > 1000


def test_find_district_rebalancing_transfers():
    # Snapshot DataFrame with 2 facilities in same district "Kenema", same product 101
    snapshot_df = pd.DataFrame([
        {
            "hf_pk": 1001,
            "name2": "Kenema Rural Clinic",
            "district": "Kenema",
            "productID": 101,
            "current_stock": 5.0,
            "consumption_roll3_mean": 60.0,  # 2 units/day -> 2.5 days of stock
            "estimated_days_of_stock": 2.5,
            "predicted_risk": 0.85
        },
        {
            "hf_pk": 1002,
            "name2": "Kenema Central Hospital",
            "district": "Kenema",
            "productID": 101,
            "current_stock": 800.0,
            "consumption_roll3_mean": 60.0,  # 2 units/day -> 400 days of stock (huge surplus)
            "estimated_days_of_stock": 400.0,
            "predicted_risk": 0.05
        },
        {
            "hf_pk": 1003,
            "name2": "Bo District Hospital",
            "district": "Bo",  # Different district -> should NOT be suggested
            "productID": 101,
            "current_stock": 900.0,
            "consumption_roll3_mean": 60.0,
            "estimated_days_of_stock": 450.0,
            "predicted_risk": 0.02
        }
    ])
    
    transfers = find_district_rebalancing_transfers(
        snapshot_df=snapshot_df,
        target_hf_pk=1001,
        target_product_id=101,
        min_surplus_days=60.0,
        reserve_days_buffer=30.0
    )
    
    assert len(transfers) == 1
    rec = transfers[0]
    assert rec["donor_hf_pk"] == 1002
    assert rec["donor_district"] == "Kenema"
    assert rec["suggested_transfer_qty"] > 0
    # Both donor and recipient should have positive, safe post-transfer stock days
    assert rec["donor_post_transfer_days"] >= 30.0
    assert rec["recipient_post_transfer_days"] >= 25.0
