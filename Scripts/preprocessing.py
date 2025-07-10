# Import necessary libraries
import pandas as pd
import numpy as np


# Define the preprocessing function
def preprocess_data(orders_df):
    # Cleaning Category Name...

    # Drop missing values in 'sku' column
    orders_df = orders_df.dropna(subset=["sku"])

    orders_df["category_name_1"].replace({"\\N": np.nan}, inplace=True)

    # Create a lookup table mapping each sku to its most common category_name_1
    category_lookup = (
        orders_df[orders_df["category_name_1"].notnull()]
        .groupby("sku")["category_name_1"]
        .agg(lambda x: x.mode().iloc[0])
    )

    # Define a function to impute missing 'category_name_1'
    def hot_deck_impute(row):
        if pd.isnull(row["category_name_1"]):
            return category_lookup.get(row["sku"], np.nan)
        return row["category_name_1"]

    # Apply the function row-wise
    orders_df["category_name_1"] = orders_df.apply(hot_deck_impute, axis=1)

    # Rename to 'Product Category'
    orders_df.rename(columns={"category_name_1": "Product Category"}, inplace=True)

    # Grouping some product categories
    orders_df["Product Category"] = orders_df["Product Category"].replace(
        {
            "Mobiles & Tablets": "Electronics & Gadgets",
            "Computing": "Electronics & Gadgets",
            "Books": "School & Education",
        }
    )

    # Cleaning Customer ID...

    orders_df = orders_df.dropna(subset=["Customer ID"])

    # Cleaning Payment Method...

    orders_df.rename(columns={"payment_method": "Payment Method"}, inplace=True)

    # Define a mapping for payment methods
    payment_map = {
        "cod": "Cash-Based",
        "cashatdoorstep": "Cash-Based",
        "ublcreditcard": "Credit/Debit Cards & Online Banking",
        "internetbanking": "Credit/Debit Cards & Online Banking",
        "bankalfalah": "Credit/Debit Cards & Online Banking",
        "Payaxis": "Credit/Debit Cards & Online Banking",
        "mygateway": "Credit/Debit Cards & Online Banking",
        "apg": "Credit/Debit Cards & Online Banking",
        "jazzwallet": "Mobile Wallets",
        "mcblite": "Mobile Wallets",
        "Easypay": "Mobile Wallets",
        "Easypay_MA": "Mobile Wallets",
        "easypay_voucher": "Vouchers & Store Credits",
        "jazzvoucher": "Vouchers & Store Credits",
        "customercredit": "Vouchers & Store Credits",
        "productcredit": "Vouchers & Store Credits",
        "marketingexpense": "Internal / Backend Adjustments",
        "financesettlement": "Internal / Backend Adjustments",
    }

    # Replace values in-place in the existing 'payment_method' column
    orders_df["Payment Method"] = orders_df["Payment Method"].replace(payment_map)

    # Remove 'Internal / Backend Adjustments' from the column
    orders_df = orders_df[
        orders_df["Payment Method"] != "Internal / Backend Adjustments"
    ]

    # Cleaning Price...

    orders_df["price"] = pd.to_numeric(orders_df["price"], errors="coerce")

    # Drop rows where price is less than 10
    orders_df = orders_df[orders_df["price"] >= 10]

    # Cleaning Discount Amount...

    orders_df.rename(columns={"discount_amount": "Discount"}, inplace=True)

    orders_df = orders_df[
        (orders_df["Discount"] >= 0)
        & (orders_df["Discount"] <= 0.5 * (orders_df["price"]))
    ]

    # Create a new column for Discount %
    orders_df["Discount %"] = (orders_df["Discount"] / (orders_df["price"])) * 100
    orders_df["Discount %"] = orders_df["Discount %"].round(1)

    # Define a function to remove outliers using the IQR
    def remove_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    # Separate the dataframe based on payment methods
    cash_df = orders_df[orders_df["Payment Method"] == "Cash-Based"]
    voucher_df = orders_df[orders_df["Payment Method"] == "Vouchers & Store Credits"]
    other_df = orders_df[
        ~orders_df["Payment Method"].isin(["Cash-Based", "Vouchers & Store Credits"])
    ]

    # Remove outliers in Discount % for cash and voucher based payment methods
    cash_df_cleaned = remove_outliers_iqr(cash_df, "Discount %")
    voucher_df_cleaned = remove_outliers_iqr(voucher_df, "Discount %")

    # Consolidate the payment method dataframes
    orders_df = pd.concat(
        [cash_df_cleaned, voucher_df_cleaned, other_df], ignore_index=True
    )

    # Cleaning Grand Total...

    for col in ["price", "qty_ordered", "Discount", "grand_total"]:
        orders_df[col] = pd.to_numeric(orders_df[col], errors="coerce")

    # Replace the mismatched grand total values with the expected values
    orders_df["Expected_Grand_Total"] = (
        orders_df["price"] * orders_df["qty_ordered"]
    ) - orders_df["Discount"]
    mismatched = orders_df["grand_total"] != orders_df["Expected_Grand_Total"]
    orders_df.loc[mismatched, "grand_total"] = orders_df.loc[
        mismatched, "Expected_Grand_Total"
    ]

    orders_df.drop(columns=["Expected_Grand_Total"], inplace=True)

    # Rename the price, qty_ordered, and grand_total columns
    orders_df.rename(columns={"price": "Unit Price"}, inplace=True)
    orders_df.rename(columns={"qty_ordered": "Quantity"}, inplace=True)
    orders_df.rename(columns={"grand_total": "Grand Total"}, inplace=True)

    # Separate the datafrrame based on payment methods
    cash_df = orders_df[orders_df["Payment Method"] == "Cash-Based"]
    voucher_df = orders_df[orders_df["Payment Method"] == "Vouchers & Store Credits"]
    wallets_df = orders_df[orders_df["Payment Method"] == "Mobile Wallets"]
    online_df = orders_df[
        orders_df["Payment Method"] == "Credit/Debit Cards & Online Banking"
    ]

    # Remove outliers in grand total across all payment methods
    cash_df_cleaned = remove_outliers_iqr(cash_df, "Grand Total")
    voucher_df_cleaned = remove_outliers_iqr(voucher_df, "Grand Total")
    wallets_df_cleaned = remove_outliers_iqr(wallets_df, "Grand Total")
    online_df_cleaned = remove_outliers_iqr(online_df, "Grand Total")

    orders_df = pd.concat(
        [cash_df_cleaned, voucher_df_cleaned, wallets_df_cleaned, online_df_cleaned],
        ignore_index=True,
    )

    # Rename the created_at column to 'Order Date'
    orders_df.rename(columns={"created_at": "Order Date"}, inplace=True)

    # Convert data types for Customer ID and Order Date
    orders_df["Customer ID"] = orders_df["Customer ID"].astype(int)
    orders_df["Order Date"] = pd.to_datetime(orders_df["Order Date"], errors="coerce")

    # Select and reorder the columns
    orders_df = orders_df[
        [
            "Customer ID",
            "Order Date",
            "Unit Price",
            "Quantity",
            "Discount %",
            "Grand Total",
            "Payment Method",
            "Product Category",
        ]
    ]

    # Sort the DataFrame by 'Customer ID' and 'Order Date'
    orders_df = orders_df.sort_values(by=["Customer ID", "Order Date"])

    # Remove duplicate rows
    orders_df = orders_df.drop_duplicates()

    return orders_df
